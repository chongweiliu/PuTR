import lap

import torch
import torch.distributed
import torch.nn.functional as F
import numpy as np

from models import build_model, build_xy_pe, build_frame_pe
class TI(object):  # TrackIndex
    TLBR = 0  # 4
    Score = 4 # 1
    ClassID = 5 # 1
    ORIG_TLBR = 6  # 4
    Hits = 10  # 1
    Age = 11  # 1
    StartFrame = 12 # 1
    EndFrame = 13 # 1
    TrackID = 14 # 1
    State = 15 # 1
    RowNum = 16


class TS(object):  # TrackState
    New = 0
    Track = 1
    Lost = 2
    Remove = 3
    
    
class RuntimeTracker:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.device = next(model.parameters()).device
        
        self.asso_thre1 = config["ASSO_THRE1"]
        self.asso_thre2 = config["ASSO_THRE2"]
        self.dim = config["DIM"]
        self.n_grid = config["PATCH_GRID"]
        self.len_sampled_feature = config["PATCH_GRID"] * config["PATCH_GRID"] * 3
        x = torch.linspace(0, self.n_grid - 1, self.n_grid, dtype=torch.float32).to(self.device)
        y = torch.linspace(0, self.n_grid - 1, self.n_grid, dtype=torch.float32).to(self.device)
        self.yv, self.xv = torch.meshgrid(x, y)
        
        self.max_nframes = config["MAX_NFRAMES"]
        self.det_score_thresh = config["DET_SCORE_THRESH"]
        self.track_score_thresh = config["TRACK_SCORE_THRESH"]
        self.miss_tolerance = config["MISS_TOLERANCE"]
        self.tracklet_score_thresh = config["TRACK_SCORE_THRESH"]
        self.new_trk_tolerance = config["NEW_TRK_TOLERANCE"]
        self.dets_iou_thresh = config["DETS_IOU_THRESH"]
        self.classwise = False

        self.ntrk_per_frame = []
        self.start_coord = np.array([1], dtype=np.int32)
        self.id_coords = {}
        self.xy_pe = build_xy_pe(config=config)
        self.frame_pe = build_frame_pe(config=config)
        self.frame_pos = self.frame_pe(torch.arange(0, self.max_nframes + 1, dtype=torch.float32, device=self.device))
        # self.emb_id_list = list(np.arange(2, self.model.n_id_embedding))
        self.tid_2_emb_id = {}
        
        
        self.bos = torch.zeros((1, config["DIM"]), dtype=torch.float32, device=self.device)
        self.trks = torch.zeros((0, TI.RowNum), dtype=torch.float32, device=self.device)
        self.trk_projected_tokens = torch.zeros((0, config["DIM"]), dtype=torch.float32, device=self.device)
        self.trk_xy_pos = torch.zeros((0, config["DIM"]), dtype=torch.float32, device=self.device)
        self.trk_frame_pos = torch.zeros((0, config["DIM"]), dtype=torch.float32, device=self.device)
        self.trk_emb_ids = torch.zeros((0, ), dtype=torch.int32, device=self.device)
      
        self.frame_id = 0
        self.id_count = 1
    
    def update(self, img, dets, orig_dets):
        # dets: [x1, y1, x2, y2, score, class_id]
        dets_remain_idx = dets[:, 4] > self.det_score_thresh
        dets = dets[dets_remain_idx]
        orig_dets = orig_dets[dets_remain_idx]
        
        img = img.permute(0, 3, 1, 2).contiguous().float().div_(255.).sub_(0.5).div_(0.2)
        det_projected_tokens = self.model.project(self.get_det_tokens(dets, img))
        cxcys = (dets[:, :2] + dets[:, 2:4]) / 2
        dets_xy_pos = self.xy_pe(cxcys[:, 0], cxcys[:, 1], img.shape[-2], img.shape[-1])
        dets_frame_pos = self.frame_pos[len(self.ntrk_per_frame)].unsqueeze(0).repeat(dets.shape[0], 1)
        # dets_emd_ids = torch.ones((dets.shape[0], ), dtype=torch.int32, device=self.device)
        

        dets_iou = self.hmiou(dets[:, :4], dets[:, :4])
        dets_iou[torch.arange(dets_iou.shape[0]), torch.arange(dets_iou.shape[0])] = 0

        
        if self.trks.shape[0] > 0:
            input_tokens = torch.cat((self.bos, self.trk_projected_tokens, det_projected_tokens), dim=0).unsqueeze(0)
            trk_ids = self.trks[:, TI.TrackID].int().tolist()
            atten_mask, col_ids = self.get_atten_mask(input_tokens, trk_ids)
            inputs_xy_pos = torch.cat((self.bos, self.trk_xy_pos, dets_xy_pos), dim=0).unsqueeze(0)
            inputs_frame_pos = torch.cat((self.bos, self.trk_frame_pos, dets_frame_pos), dim=0).unsqueeze(0)
            inputs_emd_ids = 1 #torch.cat((self.bos[0, 0:1], self.trk_emb_ids, dets_emd_ids), dim=0).unsqueeze(0).int()
            
            output_tokens = self.model(input_tokens, inputs_frame_pos, inputs_xy_pos, inputs_emd_ids, atten_mask, is_projected=True).squeeze(0)
            
            # associate
            
            cost_matrix = output_tokens @ output_tokens[self.start_coord[-1]:].T
            cost_matrix = cost_matrix.softmax(dim=1)
            
            cost_matrix_f = torch.zeros((len(col_ids), dets.shape[0]), dtype=torch.float32, device=self.device)
            
            for i, trk_id in enumerate(trk_ids):
                cost_matrix_f[i] = cost_matrix[self.id_coords[trk_id][:, -1]].mean(dim=0)
            
            # cost_matrix = output_tokens[col_ids] @ output_tokens[self.start_coord[-1]:].T
            # cost_matrix_f = cost_matrix.softmax(dim=1)
            cost_matrix_box = self.hmiou(self.trks[:, TI.TLBR:TI.TLBR + 4], dets[:, :4])
            final_cost_matrix = (cost_matrix_f + cost_matrix_box) * dets[:, 4].unsqueeze(0) * self.trks[:, TI.Score].unsqueeze(1)
            
            matches, untracks, undets = self.associate(final_cost_matrix, cost_matrix_box, cost_matrix_f, threshold_1=self.asso_thre1, threshold_2=self.asso_thre2)
            matches, untracks, undets = torch.from_numpy(matches).to(self.device), torch.from_numpy(untracks).to(self.device), torch.from_numpy(undets).to(self.device)
            
            # track
            if matches.shape[0] != 0:
                m0 = matches[:, 0]
                m1 = matches[:, 1]
                self.trks[m0, TI.Age] += 1
                t = self.trks[m0, TI.Age] >= self.new_trk_tolerance
                self.trks[matches[t, 0], TI.State] = TS.Track
                self.trks[m0, TI.EndFrame] = self.frame_id
                self.trks[m0, TI.Hits] += 1
                self.trks[m0, TI.TLBR:TI.TLBR + 6] = dets[m1, :]
                self.trks[m0, TI.ORIG_TLBR:TI.ORIG_TLBR + 4] = orig_dets[m1, :4]
                matched_det_tokens = det_projected_tokens[m1, :]
                matched_xy_pos = dets_xy_pos[m1, :]
                # matched_tid = torch.as_tensor([self.tid_2_emb_id[x] for x in self.trks[m0, TI.TrackID].int().tolist()], device=self.device, dtype=torch.int32)
          
                for i in range(len(m0)):
                    tid = self.trks[m0[i], TI.TrackID].int().item()
                    self.id_coords[tid] = np.concatenate([self.id_coords[tid], [[self.frame_id, len(self.ntrk_per_frame), i, self.start_coord[-1] + i]]], axis=0)
            else:
                matched_det_tokens = torch.zeros((0, self.dim), dtype=torch.float32, device=self.device)
                matched_xy_pos = torch.zeros((0, self.dim), dtype=torch.float32, device=self.device)
                # matched_tid = torch.zeros((0, ), dtype=torch.int32, device=self.device)
             
            #remove
            self.trks[untracks, TI.Hits] = 0
            new_idx = self.trks[:, TI.State] == TS.New
            self.trks[untracks, TI.State] = TS.Lost
            lost_idx = self.trks[:, TI.State] == TS.Lost
            rm_new_idx = torch.logical_and(new_idx, lost_idx)
            rm_lost_idx = self.frame_id - self.trks[:, TI.EndFrame] >= self.miss_tolerance
            rm_idx = torch.logical_or(rm_new_idx, rm_lost_idx)
            self.trks[rm_idx, TI.State] = TS.Remove
            
            t = self.trks[:, TI.State] != TS.Remove
            
            rm_ids = self.trks[~t, TI.TrackID]
            for rm_id in rm_ids:
                self.id_coords.pop(rm_id.int().item())
            
            self.trks = self.trks[t, :]
        else:
            matches = np.empty((0, 2), dtype=int)
            undets = torch.arange(dets.shape[0])
            matched_det_tokens = torch.zeros((0, self.dim), dtype=torch.float32, device=self.device)
            matched_xy_pos = torch.zeros((0, self.dim), dtype=torch.float32, device=self.device)
            # matched_tid = torch.zeros((0, ), dtype=torch.int32, device=self.device)
        
        dets = dets[undets, :]
        if dets_iou[undets].shape[0] != 0:
            t = torch.logical_and(
                dets[:, TI.Score] >= self.tracklet_score_thresh,
                (dets_iou[undets].max(dim=1)[0] < self.dets_iou_thresh) + (self.frame_id < 5))
        else:
            t = dets[:, TI.Score] >= self.tracklet_score_thresh
        new_trks = dets[t, :]
        
        
        self.ntrk_per_frame.append(new_trks.shape[0] + matches.shape[0])
        self.start_coord = np.concatenate([self.start_coord, [self.start_coord[-1] + self.ntrk_per_frame[-1]]])
        
        if len(self.ntrk_per_frame) > self.max_nframes:
            pop_n = self.ntrk_per_frame.pop(0)
            self.start_coord = self.start_coord[1:] - pop_n
            self.trk_projected_tokens = self.trk_projected_tokens[pop_n:]
            self.trk_xy_pos = self.trk_xy_pos[pop_n:]
            self.trk_emb_ids = self.trk_emb_ids[pop_n:]

            self.trk_frame_pos = torch.repeat_interleave(self.frame_pos[:len(self.ntrk_per_frame) - 1], torch.tensor(self.ntrk_per_frame[:-1], device=self.device), dim=0)
            for id in self.id_coords.keys():
                coords = self.id_coords[id]
                coords[:, 1] -= 1
                coords[:, 3] -= pop_n
                coords = coords[coords[:, 1] >= 0, :]
                if coords.shape[0] == 0:
                    self.id_coords.pop(id)
                    # self.emb_id_list.append(self.tid_2_emb_id.pop(id))
                else:
                    self.id_coords[id] = coords

        
        self.trk_projected_tokens = torch.cat([self.trk_projected_tokens, matched_det_tokens, det_projected_tokens[undets, :][t]], dim=0)
        self.trk_xy_pos = torch.cat([self.trk_xy_pos, matched_xy_pos, dets_xy_pos[undets, :][t]], dim=0)
        self.trk_frame_pos = torch.cat([self.trk_frame_pos, self.frame_pos[len(self.ntrk_per_frame) - 1].unsqueeze(0).repeat(self.ntrk_per_frame[-1], 1)])
        
        
        if new_trks.shape[0] != 0:
            new_trks = torch.cat([new_trks, orig_dets[undets, :4][t], torch.zeros((new_trks.shape[0], TI.RowNum - 10), device=self.device)], dim=1)

            new_trks[:, TI.Age] = 0
            new_trks[:, TI.StartFrame] = self.frame_id
            new_trks[:, TI.EndFrame] = self.frame_id
            new_trks[:, TI.State] = TS.New
            new_trks[:, TI.TrackID] = torch.arange(self.id_count, self.id_count + new_trks.shape[0], device=self.device, dtype=torch.float32)
            # new_trk_ids = new_trks[:, TI.TrackID].int()
          
            for i in range(new_trks.shape[0]):
                id = new_trks[i, TI.TrackID].int().item()
                self.id_coords[id] = np.array([[self.frame_id, len(self.ntrk_per_frame) - 1, i, self.start_coord[-2] + i + matches.shape[0]]])
                # self.tid_2_emb_id[id] = self.emb_id_list.pop(0)
                # new_trk_ids[i] = self.tid_2_emb_id[id]
            self.id_count += new_trks.shape[0]
            self.trks = torch.cat((self.trks, new_trks), axis=0)
        else:
            new_trk_ids = torch.zeros((0, ), dtype=torch.int32, device=self.device)
         
        # self.trk_emb_ids = torch.cat([self.trk_emb_ids, matched_tid, new_trk_ids], dim=0)
       
        self.frame_id += 1
        return self.trks
    
    def get_det_tokens(self, dets, img):
        if dets.shape[0] > 0:
            box = dets[:, :4]
            img_h, img_w = img.shape[-2:]
            widths = box[:, 2] - box[:, 0] + 0.5
            heights = box[:, 3] - box[:, 1] + 0.5
            
            # 计算每个网格的宽度和高度
            grid_widths = (widths / self.n_grid).view(-1, 1, 1)
            grid_heights = (heights / self.n_grid).view(-1, 1, 1)
            
            # 将网格坐标转化为每个网格的中点坐标
            grid_x = box[:, 0].view(-1, 1, 1) + (self.xv + 0.5).unsqueeze(0) * grid_widths
            grid_y = box[:, 1].view(-1, 1, 1) + (self.yv + 0.5).unsqueeze(0) * grid_heights
            
            # 将每个网格的中点坐标拼接成一个数组
            grid_coordinates = torch.stack((grid_x, grid_y), dim=3)  # (n, n_grid, n_grid, 2)
            
            # 将网格坐标转化为归一化坐标范围(-1, 1)
            grid_coordinates[:, :, :, 0] = grid_coordinates[:, :, :, 0] / img_w * 2 - 1
            grid_coordinates[:, :, :, 1] = grid_coordinates[:, :, :, 1] / img_h * 2 - 1
            
            # 使用F.grid_sample进行采样
            sampled_features = F.grid_sample(
                img, grid_coordinates.reshape(
                    1, grid_coordinates.shape[0] * grid_coordinates.shape[1], grid_coordinates.shape[2], 2),
                mode='bilinear',
                padding_mode='zeros').permute(0, 2, 3, 1).reshape(-1, self.len_sampled_feature)
            return sampled_features
        else:
            return torch.zeros((0, self.len_sampled_feature), dtype=torch.float32, device=self.device)
    
    def get_atten_mask(self, input_tokens, id_order):
        atten_mask = torch.ones((input_tokens.shape[1], input_tokens.shape[1]), dtype=torch.bool, device=self.device)
        
        col_ids = []
        for id in id_order:
            coords = self.id_coords[id]
            coords = torch.from_numpy(coords).to(self.device)
            col_ids.append(coords[-1, -1].item())
            
            # coords = coords[:, -1]
            # xc, yc = torch.meshgrid(coords, coords)
            # atten_mask.index_put_((xc.flatten(), yc.flatten()), torch.tensor(1, device=self.device, dtype=torch.bool))
        
        # atten_mask[self.start_coord[-2]:] = 0
        # atten_mask[self.start_coord[-2]:, col_ids] = 1

        
        for fidx in range(len(self.ntrk_per_frame)):
            atten_mask[self.start_coord[fidx]:self.start_coord[fidx + 1],
            self.start_coord[fidx]:self.start_coord[fidx + 1]] = 0
        
        atten_mask.tril_(diagonal=0)
        
        atten_mask[torch.arange(atten_mask.shape[1]), torch.arange(atten_mask.shape[1])] = 1
        atten_mask[:, 0] = 1
        
        atten_mask = atten_mask.unsqueeze(0).unsqueeze(0)
        return atten_mask, col_ids
    
    def associate(self, final_cost_matrix, cost_matrix_box, cost_matrix_f, threshold_1=0.1, threshold_2=0.1):
        if min(final_cost_matrix.shape) > 0:
            final_cost_matrix = final_cost_matrix.cpu().numpy()
            _, x, y = lap.lapjv(-final_cost_matrix, extend_cost=True)
            matched_indices = np.array([[y[i], i] for i in x if i >= 0])
        else:
            matched_indices = np.empty(shape=(0, 2))
        
        unmatched_detections = []
        for d in range(final_cost_matrix.shape[1]):
            if d not in matched_indices[:, 1]:
                unmatched_detections.append(d)
                
        unmatched_trackers = []
        for t in range(final_cost_matrix.shape[0]):
            if t not in matched_indices[:, 0]:
                unmatched_trackers.append(t)
        
        # filter out matched with low IOU
        matches = []
        for i, j in matched_indices:
            if (cost_matrix_box[i, j] < threshold_1) and (cost_matrix_f[i, j] < threshold_2):
                unmatched_detections.append(j)
                unmatched_trackers.append(i)
            else:
                matches.append([i, j])
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=np.int32)
        else:
            matches = np.array(matches, dtype=np.int32)
        
        return matches, np.array(unmatched_trackers, dtype=np.int32), np.array(unmatched_detections, dtype=np.int32)
    
    def hmiou(self, bboxes1, bboxes2):
        """
        Height_Modulated_IoU
        """
        # Expand dimensions to match the shape of bboxes2
        bboxes2 = bboxes2.unsqueeze(0)
        bboxes1 = bboxes1.unsqueeze(1)
        
        # Calculate the intersection area
        yy11 = torch.max(bboxes1[..., 1], bboxes2[..., 1])
        yy12 = torch.min(bboxes1[..., 3], bboxes2[..., 3])
        
        yy21 = torch.min(bboxes1[..., 1], bboxes2[..., 1])
        yy22 = torch.max(bboxes1[..., 3], bboxes2[..., 3])
        
        o = (yy12 - yy11) / (yy22 - yy21)
        
        # Calculate the union area
        xx1 = torch.max(bboxes1[..., 0], bboxes2[..., 0])
        yy1 = torch.max(bboxes1[..., 1], bboxes2[..., 1])
        xx2 = torch.min(bboxes1[..., 2], bboxes2[..., 2])
        yy2 = torch.min(bboxes1[..., 3], bboxes2[..., 3])
        
        w = torch.max(torch.zeros_like(xx2 - xx1), xx2 - xx1)
        h = torch.max(torch.zeros_like(yy2 - yy1), yy2 - yy1)
        wh = w * h
        
        # Calculate the Height-Modulated IoU
        o *= wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
                   + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)
        
        return o
    
    
