from torch.utils.tensorboard import SummaryWriter

class SummaryWriterHelper(SummaryWriter):
    def __init__(self, log_dir=None, comment="", purge_step=None, max_queue=10, flush_secs=120, filename_suffix=""):
        super().__init__(log_dir, comment, purge_step, max_queue, flush_secs, filename_suffix)
        self.tag_steps = {}
    
    def get_next_step(self, tag, global_step, auto_step) -> int:
        if tag not in self.tag_steps:
            self.tag_steps[tag] = 0
        tag_step = self.tag_steps[tag] if auto_step and not global_step else global_step
        if tag_step is not None:
            self.tag_steps[tag] = tag_step + 1
        return tag_step
        
    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False, double_precision=False, auto_step=True):
        return super().add_scalar(tag, scalar_value, self.get_next_step(tag, global_step, auto_step), walltime, new_style, double_precision)
    
    def add_text(self, tag, text_string, global_step=None, walltime=None, auto_step=True):
        return super().add_text(tag, text_string, self.get_next_step(tag, global_step, auto_step), walltime)