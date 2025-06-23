import torch

EPOCH_KEY = 'epoch'
G_MODEL_STATE_KEY = 'g_model_state'
D_MODEL_STATE_KEY = 'd_model_state'
G_OPTIMIZER_STATE_KEY = 'g_optimizer_state'
D_OPTIMIZER_STATE_KEY = 'd_optimizer_state'

def save_checkpoint(path, epoch, g_model, g_optimizer, d_model, d_optimizer):
    checkpoint_data = {
        EPOCH_KEY: epoch,
        G_MODEL_STATE_KEY: g_model.state_dict(),
        G_OPTIMIZER_STATE_KEY: g_optimizer.state_dict(),
        D_MODEL_STATE_KEY: d_model.state_dict(),
        D_OPTIMIZER_STATE_KEY: d_optimizer.state_dict(),
    }

    torch.save(checkpoint_data, path)

class CheckpointLoader:
    def __init__(self, path):
        self.checkpoint = torch.load(path, weights_only=True)
    
    def load_models(self, g_model, d_model):
        g_model.load_state_dict(self.checkpoint[G_MODEL_STATE_KEY])
        d_model.load_state_dict(self.checkpoint[D_MODEL_STATE_KEY])

    def load_optimizers(self, g_optimizer, d_optimizer):
        g_optimizer.load_state_dict(self.checkpoint[G_OPTIMIZER_STATE_KEY])
        d_optimizer.load_state_dict(self.checkpoint[D_OPTIMIZER_STATE_KEY])

    def load_epoch(self):
        return self.checkpoint[EPOCH_KEY]