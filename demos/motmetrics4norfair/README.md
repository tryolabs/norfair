# Compute metrics and obtain your predictions from MOTchallenge detections

You can find MOTchallenge datasets (detections and ground truth files) in https://motchallenge.net

Click on the following link to download MOT17 challenge data, including folder with pictures for each video (5.5 GB)
    https://motchallenge.net/data/MOT17.zip
Alternatively, you can download the same challenge without the corresponding images (9.7 MB)
    https://motchallenge.net/data/MOT17Labels.zip

You can download the py-motmetrics running the following command on terminal:
    pip install motmetrics

The official py-motmetrics repository can be found in:
https://github.com/cheind/py-motmetrics

To run motmetrics4norfair, run the following command on terminal:
    python motmetrics4norfair.py <files_to_process>

After that, the metrics results obtained by comparing your predictions with the ground truth data will be printed on terminal

In case you use the optional command --save_pred or --make_video or --save_metrics, you must specify an <output_path> when calling motmetrics4norfair, as in:
    python motmetrics4norfair.py <files_to_process> <output_path> --save_metrics --make_video

Be sure that for each <video_file> in <Name_of_video_files> you have the following files before calling motmetrics4norfair:
    <video_file>/det/det.txt

Also be sure you have the ground truth data:
    <video_file>/gt/gt.txt

Take into account that to make the output video you need to have the folder <video_file>/img1/ containing the corresponging pictures of the videos being proccessed.