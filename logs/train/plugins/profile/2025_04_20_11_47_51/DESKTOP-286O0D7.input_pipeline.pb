  *fffffݲ@������@)      p=2o
8Iterator::Model::MaxIntraOpParallelism::ShuffleAndRepeatu�^@!cɨ��GG@)`vO*@1���F@:Preprocessing2�
eIterator::Model::MaxIntraOpParallelism::ShuffleAndRepeat::Prefetch::MapAndBatch::FlatMap[0]::TFRecord�5�;NQ@!ߢ-�D@)�5�;NQ@1ߢ-�D@:Advanced file read2�
XIterator::Model::MaxIntraOpParallelism::ShuffleAndRepeat::Prefetch::MapAndBatch::FlatMap�Ǻ�@!ESa&��F@)���S��?14�}���@:Preprocessing2�
OIterator::Model::MaxIntraOpParallelism::ShuffleAndRepeat::Prefetch::MapAndBatch{�/L�
�?!�eI���@){�/L�
�?1�eI���@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::ShuffleAndRepeat::PrefetchU���N@�?!X*|K	@)U���N@�?1X*|K	@:Preprocessing2F
Iterator::Model@�߾�@!bJ���I@)�D����?1� {ں@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?W[��l@!���NaH@)�L�J��?1#����@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb��No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.