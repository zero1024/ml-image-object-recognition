package poa.ml.image.objects.recognition.labeler

import poa.ml.image.objects.recognition.model.Sample
import java.awt.image.BufferedImage

interface ImageSampleLabeler {

    suspend fun label(samples: List<Sample>, sourceImage: BufferedImage): List<Pair<Sample, Boolean>>

}
