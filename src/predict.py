"""
predict.py用于使用训练好的model进行EE任务

"""
import os
from config import Config



# 获取最后保存的 Model
def get_latest_saved_model(model_dir):
    all_checkpoint_dir = [os.path.join(model_dir, checkpoint_dir) for checkpoint_dir in os.listdir(model_dir)]

    latest_checkpoint_dir = max(all_checkpoint_dir, key=os.path.getmtime)

    latest_model = os.path.join(latest_checkpoint_dir,'model.pt')
    assert(os.path.isfile(latest_model))



if __name__ == '__main__':
    model_dir = Config.saved_trigger_extractor_dir
    latest = get_latest_checkpoint_dir(model_dir)
    get_latest_saved_model(model_dir)
    # print(latest)