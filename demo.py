import norfair

video = norfair.Video(input_path="/home/lalo/data/videos/in/trr_cut_short.mp4")
# tracker = norfair.Tracker()
# detector = SomeExternalDetector()

for frame in video:
    # detections = detector(frame)
    # converted_detections = convertor(detections)
    # estimates = tracker.update(converted_detections, dt=3)
    # norfair.draw_midpoint(frame, estimates)
    # norfair.draw_pose(frame, converted_detections, colors.blue)
    video.write(frame)
    video.show(frame, downsample_ratio=4)

# video.save()  # DELETE?
