package poa.ml.image.objects.recognition

import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.awt.Color
import java.awt.image.BufferedImage
import javax.swing.JButton

class ImageSampleLabeler {

    suspend fun label(samples: List<Sample>, sourceImage: BufferedImage): List<Pair<Sample, Boolean>> {
        return withContext(Dispatchers.Default) {
            val res = mutableListOf<Pair<Sample, Boolean>>()
            var cur = 0
            while (cur < samples.size) {
                val sample = samples[cur++]
                val label = showDialogAndLabel(highlightSample(sourceImage, sample))
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


private fun highlightSample(sourceImage: BufferedImage, sample: Sample): BufferedImage {
    val height = sample.image.height
    val width = sample.image.width
    val res = sourceImage.clone()
    for (x in sample.x until sample.x + width) {
        res.setRGB(x, sample.y, Color.WHITE.rgb)
    }
    for (x in sample.x until sample.x + width) {
        res.setRGB(x, sample.y + height - 1, Color.WHITE.rgb)
    }
    for (y in sample.y until sample.y + height) {
        res.setRGB(sample.x, y, Color.WHITE.rgb)
    }
    for (y in sample.y until sample.y + height) {
        res.setRGB(sample.x + width - 1, y, Color.WHITE.rgb)
    }

    return res
}
