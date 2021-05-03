package poa.ml.image.objects.recognition.labeler

import poa.ml.image.objects.recognition.model.Sample
import java.awt.image.BufferedImage

class FixedImageSampleLabeler(private val label: Boolean) : ImageSampleLabeler {

    override suspend fun label(samples: List<Sample>, sourceImage: BufferedImage): List<Pair<Sample, Boolean>> {
        return samples.map { it to label }
    }


}
