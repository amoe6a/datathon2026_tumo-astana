from scripts.evaluate_qwen_rag import summarize


def test_even_sample_timing_median_and_failed_video_denominator():
    rows=[{'status':'done','expected':1,'predicted':1,'seconds':2.,'nearest_anchor_error_sec':1.},
          {'status':'done','expected':1,'predicted':1,'seconds':4.,'nearest_anchor_error_sec':9.},
          {'status':'done','expected':0,'predicted':0,'seconds':1.},
          {'status':'failed','expected':0,'seconds':10.}]
    result=summarize(rows)
    assert result['median_nearest_anchor_error_sec']==5.
    assert result['nearest_anchor_evaluated_videos']==2
    assert result['median_warm_seconds']==3.
    assert result['processed']==3 and result['failed']==1
    assert result['accuracy_including_failures']==.75
    assert result['max_warm_seconds']==10.
