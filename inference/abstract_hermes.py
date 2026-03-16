import torch

class Abstract_Hermes:
    kv_cache = None

    def __init__(self, processor, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size, kv_size):
        self.processor = processor
        self.n_frame_tokens = n_frame_tokens
        self.init_prompt_ids = init_prompt_ids
        self.n_local = n_local
        self.topk = topk
        self.chunk_size = chunk_size
        self.kv_size = kv_size
        self.last_encoded_frames = 0
        self.visual_start_idx = 14
        self.conv_history = []
        
    def clear_cache(self):
        self.kv_cache = None
        self.conv_history = []
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def encode_init_prompt(self):
        if not isinstance(self.init_prompt_ids, torch.Tensor):
            self.init_prompt_ids = torch.as_tensor(self.init_prompt_ids, device=self.device)
        output = self.language_model(input_ids=self.init_prompt_ids, use_cache=True, return_dict=True)
        self.kv_cache = output.past_key_values
        self.visual_start_idx = self.kv_cache[0][0].shape[2]

    def get_prompt(self, query, mc=False):
        prompt = f"\n{query}<|im_end|><|im_start|>assistant\n"
        
        if mc:
            prompt += 'Best option: ('
        return prompt

    def get_video_features(self, pixel_values_videos):
        batch_size, frames, channels, height, width = pixel_values_videos.shape
        pixel_values_videos = pixel_values_videos.view(batch_size * frames, channels, height, width)
        video_features = self.vision_tower(pixel_values_videos, output_hidden_states=True)
        selected_video_feature = video_features.hidden_states[self.config.vision_feature_layer]
        if self.config.vision_feature_select_strategy == "default":
            selected_video_feature = selected_video_feature[:, 1:]
        elif self.config.vision_feature_select_strategy == "full":
            selected_video_feature = selected_video_feature
        video_features = self.multi_modal_projector(selected_video_feature)
        video_features = self.apply_pooling(video_features)
        #video_features = self.frame_filtering(video_features)
        frames_after_merge = video_features.shape[0]
        video_features = video_features.reshape(batch_size, frames_after_merge * video_features.shape[1], -1)  # (B, Nv*196, D)
        return video_features

    def get_gpu_memory_usage_gb(self):
        return torch.cuda.max_memory_allocated() / (1024**3)