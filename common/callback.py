import torch
from tqdm import tqdm, trange
from tqdm._utils import _term_move_up
prefix = _term_move_up() + '\r'

"""
An empty Callback object to be called inside a Trainer (see trainer.py)

Callback affect the trainer they are associated with, or provide log infos, or anything you can think of.
Inside a Trainer, they can be called at three points :
- At the end of a forward/backward pass
- At the end of an training epoch
- At the end of a testing epoch
"""
class Callback:
    def __init__(self, *args, **kwargs):
        pass

    def callOnEndForward(self, trainer):
        pass

    def callOnEndTrain(self, trainer):
        pass

    def callOnEndTest(self, trainer):
        pass
    

"""
A Specific Callback that is responsible for calling the logger of the trainer.
Registers metrics of the training inside a .log file
"""
class LoggerCB(Callback):

    def callOnEndTrain(self, trainer):
        if trainer.logger is not None:
            trainer.logger.update({"epoch" :  trainer.metrics["epoch"]})
            trainer.logger.update({"time" :  trainer.metrics["epoch_time"]})
            trainer.logger.update({"train_loss" : trainer.metrics["train_loss"]})
            for i,k in enumerate(trainer.topk):
                trainer.logger.update({"train_acc(top{})".format(k) : trainer.metrics["train_acc"][i]})

    def callOnEndTest(self, trainer):
        if trainer.logger is not None:
            trainer.logger.update({"test_loss" : trainer.metrics["test_loss"]})
            for i,k in enumerate(trainer.topk):
                trainer.logger.update({"test_acc(top{})".format(k) : trainer.metrics["test_acc"][i]})
            trainer.logger.log()



"""
A Specific Callback responsible for saving the model currently in training into a file
"""
class CheckpointCB(Callback):

    def callOnEndTest(self, trainer):
        epoch = trainer.metrics["epoch"]
        if trainer.checkpointFreq != 0 and epoch%trainer.checkpointFreq == 0:
            name = "{}_epoch{}_acc{:d}.model".format(trainer.name, epoch, int(100*trainer.metrics["test_acc"][0]))
            torch.save(trainer.model.state_dict(), name)


"""
A Specific callback responsible for displaying the progress with a tqdm progress bar
"""    
class TqdmCB(Callback):
    
    def callOnEndForward(self, trainer):
        
        epoch = trainer.metrics["epoch"]
        bidx = trainer.metrics["batch_idx"]
        lr = trainer.metrics["lr"]
        train_loss = trainer.metrics["train_loss"]/(1+bidx)
        #test_loss = trainer.metrics["test_loss"]
        test_loss = trainer.metrics["CE"]

        train_acc = trainer.metrics["train_acc"]/(1+bidx)
        test_acc = trainer.metrics["test_acc"]

        tqdm_log = prefix+"Epoch {}, LR: {:.1E}, Train_Loss: {:.3f}, Test_loss: {:.3f}, ".format(
                               epoch,       lr,            train_loss,         test_loss)

        for i,k in enumerate(trainer.topk):
            tqdm_log += "Train_acc(top{}): {:.3f}, Test_acc(top{}): {:.3f}".format(
                                      k, train_acc[i],         k,   test_acc[i])

        tqdm.write(tqdm_log)

    def callOnEndTrain(self, trainer):
        print()
        