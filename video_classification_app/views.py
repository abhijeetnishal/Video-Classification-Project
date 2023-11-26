from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import numpy as np
import torch
from pytorchvideo.transforms import UniformTemporalSubsample
from pytorchvideo.data.encoded_video import EncodedVideo

@csrf_exempt
def classify_video(request):
    if request.method == 'POST':
        # video_file = request.FILES.get('video')

        # Your model inference code here
        # Load the video.
        video = EncodedVideo.from_path("/home/abhijeet/djangoproject/video_classification_project/video_classification_app/football.mp4")
        video_data = video.get_clip(start_sec=4.0, end_sec=8.0)["video"]

        # Sub-sample a fixed set of frames and convert them to a NumPy array.
        num_frames = 16
        subsampler = UniformTemporalSubsample(num_frames)
        subsampled_frames = subsampler(video_data)
        video_data_np = subsampled_frames.numpy().transpose(1, 2, 3, 0)

        processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

        # Preprocess the video frames.
        inputs = processor(list(video_data_np), return_tensors="pt")

        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Model predicts one of the 400 Kinetics 400 classes
        predicted_label = logits.argmax(-1).item()
        predicted_video_data = model.config.id2label[predicted_label]

        # Example response
        response_data = {
            'predicted_label': predicted_label,
            'predicted_video_data': predicted_video_data
        }

        return JsonResponse(response_data)

    return JsonResponse({'error': 'Invalid request method'})
