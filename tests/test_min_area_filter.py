from rtsm.models.fastsam.helper import FastSAMHelper
from rtsm.utils.min_area_filter import min_area_filter_idx
from rtsm.utils.ann import prepare_ann

if __name__ == "__main__":
    helper = FastSAMHelper(model_path="model_store/fastsam/FastSAM-x.pt", device="cuda")
    ann = helper.run_fastsam_on_image("test_dataset/rgb/1754989062.627478.png")
    ann_bool = prepare_ann(ann)
    keep, bboxes = min_area_filter_idx(ann_bool, 4000)
    print(len(keep))
    print(len(bboxes))

