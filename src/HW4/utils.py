import torch

EPOCH_KEY = 'epoch'
G_MODEL_STATE_KEY = 'g_model_state'
D_MODEL_STATE_KEY = 'd_model_state'
G_OPTIMIZER_STATE_KEY = 'g_optimizer_state'
D_OPTIMIZER_STATE_KEY = 'd_optimizer_state'

def load_checkpoint(path, g_model, g_optimizer, d_model, d_optimizer):
    checkpoint = torch.load(path, weights_only=True)
    g_model.load_state_dict(checkpoint[G_MODEL_STATE_KEY])
    d_model.load_state_dict(checkpoint[D_MODEL_STATE_KEY])
    g_optimizer.load_state_dict(checkpoint[G_OPTIMIZER_STATE_KEY])
    d_optimizer.load_state_dict(checkpoint[D_OPTIMIZER_STATE_KEY])
    return checkpoint[EPOCH_KEY]

def save_checkpoint(path, epoch, g_model, g_optimizer, d_model, d_optimizer):
    checkpoint_data = {
        EPOCH_KEY: epoch,
        G_MODEL_STATE_KEY: g_model.state_dict(),
        G_OPTIMIZER_STATE_KEY: g_optimizer.state_dict(),
        D_MODEL_STATE_KEY: d_model.state_dict(),
        D_OPTIMIZER_STATE_KEY: d_optimizer.state_dict(),
    }

    torch.save(checkpoint_data, path)