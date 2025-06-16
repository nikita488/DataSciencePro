from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ChatTrainLogEntry:
    def __init__(self, epoch, total_loss, true_set, pred_set):
        self.epoch = epoch
        self.total_loss = total_loss
        self.true_set = true_set
        self.pred_set = pred_set

class ChatTrainLogger:
    def __init__(self, experiment_name, num_epochs):
        self.writer = SummaryWriter(log_dir=f'runs/{experiment_name}')
        self.num_epochs = num_epochs

    def __enter__(self):
        return self
    
    def __exit__(self, type, value, traceback):
        self.writer.close()

    def log(self, entry):
        epoch = entry.epoch

        accuracy = accuracy_score(entry.true_set, entry.pred_set)
        precision = precision_score(entry.true_set, entry.pred_set, average='weighted', zero_division=0)
        recall = recall_score(entry.true_set, entry.pred_set, average='weighted', zero_division=0)
        f1 = f1_score(entry.true_set, entry.pred_set, average='weighted', zero_division=0)

        self.writer.add_scalar('Loss/Total', entry.total_loss, epoch)
        self.writer.add_scalar('Metrics/Accuracy', accuracy, epoch)
        self.writer.add_scalar('Metrics/Precision', precision, epoch)
        self.writer.add_scalar('Metrics/Recall', recall, epoch)
        self.writer.add_scalar('Metrics/F1 Score', f1, epoch)

        if (epoch+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {entry.total_loss:.4f}')