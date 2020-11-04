# Compute metrics and obtain your predictions from MOTchallenge detections

You can download the MOTchallenge data (detections and ground truth files) from https://motchallenge.net/data/MOT17/

To compare your predictions with the ground truth files, you can download the py-motmetrics repository from
https://github.com/cheind/py-motmetrics

To run motmetrics4norfair, run the following command on terminal:
    python <path_to_our_demo>/motmetrics4norfair.py <input_path> <output_path> <Name_of_video_files>

After that, you should have a txt file called "metrics.txt" in your <output_path> folder. This file will have the resultant metrics obatined by comparing your predictions with the ground truth data.

Optional arguments when calling our demo:
    save_pred : To save you predictions in <output_path>/predictions/
    make_video : To make a video with your predictions and detections in <output_path>/videos/
    show_metrics : To display your metrics on terminal 
    all_videos : To process every every folder in <input_path> such that the folder name starts with the "MOT" prefix. 
                When using all_videos argument, any <video_file> declared in <Name_of_video_files> argument will be treated as an exception, 
                i.e: those folders in <Name_of_video_files> will be discarded, and every folder that is not specified in <Name_of_video_files> will be processed.
                When using all_videos, the names of the folders in <input_path> that will be processed should all start with the "MOT" prefix, take "MOT17-02-FRCNN" as an example of a valid folder name to process. Those folders that don't start with "MOT" prefix will be discarded.

Be sure that for each <video_file> in <Name_of_video_files> you have the following files before calling motmetrics4norfair:
    <input_path>/<video_file>/det/det.txt

Also be sure you have the ground truth data:
    <input_path>/<video_file>/gt/gt.txt

After you run motmetrics4norfair with the argument save_pred, you can re calculate your metrics running:
    python -m motmetrics.apps.eval_motchallenge <input_path> <output_path>/predictions/

Take into account that to make the output video you need to have the folder <input_path>/<video_file>/img1/ containing the corresponging pictures of the videos being proccessed.