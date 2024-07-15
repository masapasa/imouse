VIDEO_DIR_PATH = "/kaggle/input/videos-mouse/"
IMAGE_DIR_PATH = "/kaggle/input/images-mouse/"
SAMPLE_SIZE = 16
SAMPLE_GRID_SIZE = (4, 4)
SAMPLE_PLOT_SIZE = (16, 16)
from autodistill.detection import CaptionOntology

ontology=CaptionOntology({
    "mouse": "mouse",
    "water": "water"
})
DATASET_DIR_PATH = f"{HOME}/dataset"
from autodistill_grounded_sam import GroundedSAM

base_model = GroundedSAM(ontology=ontology)
dataset = base_model.label(
    input_folder=IMAGE_DIR_PATH,
    extension=".png",
    output_folder=DATASET_DIR_PATH)