# README

## What is OCR?

OCR (Optical Character Recognition) is a technology that allows a computer to read text from images and convert it into editable text. 
It is commonly used for documents, receipts, invoices, books, forms, and ID cards.

## Which OCR Library Did I Use and Why?

I used "EasyOCR" for this project because it is easy to use, provides good accuracy, and supports many languages. 
It worked well for reading text from different types of images without requiring a complicated setup.

## What Preprocessing Techniques Improved the Results?

To improve OCR accuracy, I applied the following preprocessing techniques:

* Grayscale Conversion: Converts the image into shades of gray, making the text easier to detect.
* Histogram Equalization: Improves the contrast between the text and the background, making the text clearer.

These preprocessing steps helped improve the OCR results on some images.

## Challenges Faced While Extracting Text

I did not face any major challenges while completing this project. EasyOCR was able to extract text successfully from the images I tested. 
The main issue I noticed was that the first time EasyOCR loaded, it took a little longer because the OCR model had to initialize. 
After that, the text extraction worked smoothly.
