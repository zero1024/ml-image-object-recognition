package poa.ml.image.objects.recognition.labeler

import poa.ml.image.objects.recognition.model.Sample

class FixedImageSampleLabeler {

    fun label(samples: List<Sample>, label: Boolean): List<Pair<Sample, Boolean>> {
        return samples.map { it to label }
    }


}
