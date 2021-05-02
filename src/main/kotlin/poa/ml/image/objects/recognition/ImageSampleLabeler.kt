package poa.ml.image.objects.recognition

import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.awt.image.BufferedImage
import javax.swing.JButton

class ImageSampleLabeler {

    suspend fun label(samples: List<Sample>): List<Pair<Sample, Boolean>> {
        return withContext(Dispatchers.Default) {
            val res = mutableListOf<Pair<Sample, Boolean>>()
            var cur = 0
            while (cur < samples.size) {
                val sample = samples[cur++]
                val label = showDialogAndLabel(sample.image)
                res.add(sample to label)
            }
            res
        }
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
