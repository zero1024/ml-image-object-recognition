package poa.ml.image.objects.recognition.labeler

import kotlinx.coroutines.CompletableDeferred
import poa.ml.image.objects.recognition.highlightArea
import poa.ml.image.objects.recognition.model.Sample
import poa.ml.image.objects.recognition.showImage
import java.awt.image.BufferedImage
import javax.swing.JButton

class ManualImageSampleLabeler {

    suspend fun label(samples: List<Sample>, sourceImage: BufferedImage): List<Pair<Sample, Boolean>> {
        val res = mutableListOf<Pair<Sample, Boolean>>()
        var cur = 0
        while (cur < samples.size) {
            val sample = samples[cur++]
            val label = showDialogAndLabel(highlightArea(sourceImage, sample.toArea()))
            res.add(sample to label)
        }
        return res
    }

    private suspend fun showDialogAndLabel(image: BufferedImage): Boolean {
        val res = CompletableDeferred<Boolean>()
        showImage(image) { frame ->
            val yes = JButton("Yes")
                .apply {
                    addActionListener {
                        res.complete(true)
                        frame.dispose()
                    }
                }
            val not = JButton("No")
                .apply {
                    addActionListener {
                        res.complete(false)
                        frame.dispose()
                    }
                }
            frame.contentPane.add(yes)
            frame.contentPane.add(not)
        }
        return res.await()
    }

}
