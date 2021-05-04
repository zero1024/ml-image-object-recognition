package poa.ml.image.objects.recognition

import kotlinx.coroutines.CompletableDeferred
import poa.ml.image.objects.recognition.labeler.DrawableJLabel
import poa.ml.image.objects.recognition.model.Sample
import java.awt.Rectangle
import java.awt.image.BufferedImage
import javax.swing.JButton

class ImageCutter(private val minSize: Int) {

    suspend fun cut(image: BufferedImage, frameName: String = "Cut"): Pair<Sample, List<Sample>> {
        val res = CompletableDeferred<Rectangle>()
        val drawableJLabel = DrawableJLabel(image.clone(),minSize)
        showJLabel(drawableJLabel) { frame ->
            frame.title = frameName
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



