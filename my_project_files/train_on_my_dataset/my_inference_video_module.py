import cv2
import mmcv

from mmdet.apis import inference_detector, init_detector


def parse_video(video, architecture_config, checkpoints, result_path='', show=False, wait_time=1.0, score_thr=0.3,
                device='cuda:0'):

    assert result_path or show, ('Please specify at least one operation (save/show the video)')

    model = init_detector(architecture_config, checkpoints, device=device)

    video_reader = mmcv.VideoReader(video)
    video_writer = None
    if result_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            result_path, fourcc, video_reader.fps,
            (video_reader.width, video_reader.height))

    for frame in mmcv.track_iter_progress(video_reader):
        result = inference_detector(model, frame)
        frame = model.show_result(frame, result, score_thr=score_thr)
        if show:
            cv2.namedWindow('video', 0)
            mmcv.imshow(frame, 'video', int(wait_time))
        if result_path:
            video_writer.write(frame)

    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

