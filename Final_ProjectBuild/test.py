#from ultralytics import YOLO

#model = YOLO('Final_Project_Build_JetsonOrinNano/pytorch_model/best100.pt')
#model.eval()

from ultralytics import YOLO

# Load the model
model = YOLO('Final_Project_Build_JetsonOrinNano/pytorch_model/best100.pt')

# Set the dataset configuration path
data_path = 'Final_Project_Build_JetsonOrinNano/pytorch_model/racconnall.v1i.yolov8/data.yaml'

# Evaluate the model
results = model.val(data=data_path)

# Print the evaluation results
print(results)

#results = model.track(source=2, show=True)