def get_tracking_end_frame(exit_frame=None, max_frame=None, analysis_end_frame=None):
    """Return the frame that should end analysis tracking.

    We deliberately do not stop tracking as soon as the bee exits the arena.
    Tracking should continue until the video ends, unless the user explicitly
    marks a later analysis end frame.
    """
    if analysis_end_frame is not None:
        return int(analysis_end_frame)
    if max_frame is None:
        return int(exit_frame) if exit_frame is not None else 0
    return int(max_frame)
