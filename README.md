# Super-Resolution-Image-Generation
Super-Resolution Image Generation using GANs

This repository contains the code, resources, and documentation for my Bachelor's thesis at **Amirkabir University of Technology (Tehran Polytechnic)**.

---

## **Table of Contents**
- [About the Project](#about-the-project)
- [Dataset](#dataset)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Acknowledgments](#acknowledgments)

---

## **About the Project**
—This Project outlines the Super-Resolution Image
 Generation using GANs. Single-image super-resolution facilitates
 the enhancement of low-resolution images to higher resolutions.
 Many recent advancements in generative adversarial networks
 (GANs) exhibit promising results with limited data. This report,
 by reviewing related previous works on this specific subject, not
 only implements a model but it also introduces the elements of
 the model. The dataset contains both low-resolution and high
resolution pictures to train the generative part of the model and
 can give the discriminator of the model, the perspective on how to
 punish the model in terms of having better accuracy in training.

**University:** Toronto Metropolitan University
**Faculty:** Computer Engineering    
**Author:** Ramtin Asgarianamiri  
## **Dataset**
In the context of Super-Resolution Image Generation, pro
viding the model with a suitable dataset can be fulfilled
 through different approaches. In selecting an appropriate
 dataset, it’s essential to have a set of images with low-quality
 and high-quality versions. This collection of images, called a
 dataset, plays a big role in how well your model performs
 and how useful it is. The Dataset which specifically has been
 introduced to be used and imported in models for the high
resolution problem is called DIV2K. [15]
 DIV2K is a popular single-image super-resolution dataset
 that contains 1,000 images with different scenes and is
 split into 800 for training, 100 for validation, and 100 for
 testing. It was collected for NTIRE2017 and NTIRE2018
 Super-Resolution Challenges to encourage research on im
age super-resolution with more realistic degradation. This
 dataset contains low-resolution images with different types of
 degradations. Apart from the standard bicubic downsampling,
 several types of degradations are considered in synthesizing
 low-resolution images for different tracks of the challenges.Track 2 of NTIRE 2017 contains low-resolution images with
 unknown x4 downscaling. Track 2 and track 4 of NTIRE 2018
 correspond to realistic mild ×4 and realistic wild ×4 adverse
 conditions, respectively. Low-resolution images under realistic
 mild x4 settings suffer from motion blur, Poisson noise, and
 pixel shifting. Degradations under realistic wild x4 setting are
 further extended to be of different levels from image to image.
 In this project, we have utilized the dataset and aimed to
 enhance the resolution of the image by upsampling. Figure 1
 is an example that shows both high and low-resolution images
 from the dataset
 

## **Project Structure**

 Generator
 The process starts with a low-resolution image as input. This
 could be an image that is, for this project, 128x128 pixels. The
 input image is first passed through an initial processing step.
 This step is designed to extract basic features from the low
resolution image. These features might include edges, basic
 shapes, and general structures present in the image. Next, the
 image goes through a series of blocks called ResidualBlocks.
 Each ResidualBlock is like a mini-network within the Gen
erator. These blocks learn and refine detailed features of the
 image. They capture intricate patterns, textures, and nuances
 that make up the image content. The information from each
 ResidualBlock is then combined with the original image. After
 going through multiple ResidualBlocks, the image features
 are further refined. Another convolutional block, often called
 conv, works to enhance the image quality. It sharpens edges,
 enhances textures, and improves overall visual appearance.
 This stage is crucial for making the generated image look
 more realistic and detailed. The refined image features are then
 passed through an upsampling process. Upsampling increases
 the size and resolution of the image. This step helps to restore
 lost details that were downscaled in the low-resolution input.
 By increasing the image dimensions, it creates a larger canvas
 for finer details to be added. The upscaled image features
 are fed into a final convolutional layer. This layer acts as a
 mapping from the enhanced features to the high-resolution
 output. It generates the final image that is now much larger
 and more detailed than the input. The output size is often
 chosen to match the desired high resolution, our output image
is 256x256 pixels. Before the final image is produced, an
 activation function like sigmoid is applied. This function scales
 the pixel values to a range between 0 and 1. It ensures that the
 output image has valid pixel intensities and is ready for display
 or further processing. The result is a high-resolution image that
 is detailed, realistic, and suitable for various applications.
 C. Discriminator
 Discriminator acts like a critic in the neural network world,
 looking at images and deciding if they’re real or fake. When
 an image is given to a Discriminator, it goes through a series
 of convolutional blocks (ConvBlocks) initially. These blocks
 help the Discriminator learn about the shapes, edges, and
 patterns in the image, understanding its basic features. After
 this analysis, the learned features are then passed through a
 Multi-Layer Perceptron (MLP). The MLP takes these learned
 features and makes the final call: Is this image close to
 the test image, like a high-resolution photo, or not? This
 process of breaking down the image into features and using
 the MLP for judgment helps the Discriminator decide the
 authenticity of the image. Crucially, Discriminator plays a
 pivotal role in Generative Adversarial Networks (GANs). It
 aids the Generator in improving its images by providing
 feedback. By training Generator to fool Discriminator, we
 get progressively better and more realistic images over time.
 Essentially, the Discriminator is the discerning eye that helps
 guide the Generator toward creating images that look nearly
 indistinguishable from real ones, expanding the realm of
 artificial creativity within neural networks.
## **Usage**
 The model was built by including all the specified architec
tural details mentioned earlier. The dataset was used to train
 the model, with the following hyperparameters set: epoch =
 30, batch size = 6.
 In Figure 4-6, we can see the differences between the low
resolution image and the predicted output from the model,
 as well as the original high-resolution image in the dataset.
 This comparison highlights the model’s progress in identifying
 edges and finer details, thus improving the resolution of the
 input. The learning process occurs through adjustments made
 by the discriminator, demonstrating how the model gradually
 improves its ability to discern intricate features over time.
 Discriminator and generation loss have been extracted from the
 training for each epoch so it could be an element of discrimi
nation in terms of model evaluation.[Figure 7] In a Generative
 Adversarial Network (GAN), the Generator Loss measures
 how effectively the generator can deceive the discriminator by
 encouraging the production of samples that closely resemble
 real data, often computed using binary cross-entropy or mean
 squared error loss. Conversely, the Discriminator Loss assesses
 the discriminator’s ability to differentiate between real and
 fake samples, penalizing misclassifications with loss functions
 like binary cross-entropy or hinge loss. Through iterative
 training, the generator minimizes its loss while maximizing
 the discriminator’s loss, fostering an adversarial dynamic
 ## **Acknowledgement**
 The LOss and a test sample of the network.
![loss](https://github.com/user-attachments/assets/4539fdd2-90c6-4a49-b5d7-e3b3af8e4993)

![test](https://github.com/user-attachments/assets/a12032df-8b0b-4857-bd6e-bf31e3100bff)


