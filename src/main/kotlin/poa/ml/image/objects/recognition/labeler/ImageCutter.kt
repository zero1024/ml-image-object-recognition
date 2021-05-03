package poa.ml.image.objects.recognition.labeler

import kotlinx.coroutines.CompletableDeferred
import poa.ml.image.objects.recognition.clone
import poa.ml.image.objects.recognition.model.Sample
import poa.ml.image.objects.recognition.showImage
import poa.ml.image.objects.recognition.showJLabel
import java.awt.Rectangle
import java.awt.image.BufferedImage
import javax.swing.JButton

class ImageCutter {

    suspend fun cut(image: BufferedImage): Pair<Sample, List<Sample>> {
        val res = CompletableDeferred<Rectangle>()
        val drawableJLabel = DrawableJLabel(image.clone())
        showJLabel(drawableJLabel) { frame ->
            val yes = JButton("Cut")
                .apply {
                    addActionListener {
                        res.complete(drawableJLabel.getRectangle())
                        frame.dispose()
                    }
                }
            frame.contentPane.add(yes)
        }

        val r = res.await()

        val highlightedPart = image.getSubimage(r.x, r.y, r.width, r.height)
        val remainedPart1 = image.getSubimage(0, 0, image.width, r.y)
        val remainedPart2 = image.getSubimage(r.x + r.width, 0, image.width - (r.x + r.width), image.height)
        val remainedPart3 = image.getSubimage(0, r.y + r.height, image.width, image.height - (r.y + r.height))
        val remainedPart4 = image.getSubimage(0, 0, r.x, image.height)

        return Sample(r.x, r.y, highlightedPart) to listOf(
            Sample(0, 0, remainedPart1),
            Sample(r.x + r.width, 0, remainedPart2),
            Sample(0, r.y + r.height, remainedPart3),
            Sample(0, 0, remainedPart4)
        )
    }

}



