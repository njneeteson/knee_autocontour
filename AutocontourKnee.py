import SimpleITK as sitk

class AutocontourKnee:
    """
    A class for computing the periosteal and endosteal masks for a knee
    HR-pQCT image using only thresholding and morphological operations.

    Attributes
    ----------
    in_value : int

    peri_s1_sigma : float

    peri_s1_support : int

    peri_s1_lower : float

    peri_s1_upper : float

    peri_s2_sigma : float

    peri_s2_support : int

    peri_s2_lower : float

    peri_s2_upper : float

    peri_s2_radius : int

    peri_s3_sigma : float

    peri_s3_support : int

    peri_s3_lower : float

    peri_s3_upper : float

    peri_s3_radius : int

    peri_s4_open_radius : int

    peri_s4_close_radius : int

    DEFAULT_MAX_ERROR : float
        Needed for the procedural interface of the sitk gaussian filter.

    USE_SPACING : bool
        Needed for the procedural interface of the sitk gaussian filter.

    Methods
    -------
    _gaussian_and_threshold(img, sigma, support, lower, upper)

    _get_largest_connected_component(img)

    _inert_binary_image(img)

    _close_with_connected_components(img, radius)

    get_periosteal_mask(img)

    get_endosteal_mask(img, peri)

    get_masks(img)

    """

    def __init__(
        self,
        in_value = 127,
        peri_s1_sigma = 1.5,
        peri_s1_support = 1,
        peri_s1_lower = 400, # was 350 mgHA/ccm
        peri_s1_upper = 10000, # very high value
        peri_s1_radius = 35,
        peri_s2_sigma = 1.5,
        peri_s2_support = 1,
        peri_s2_lower = 350, # was 250 mgHA/ccm
        peri_s2_upper = 10000,
        peri_s2_radius = 10,
        peri_s3_sigma = 1.5,
        peri_s3_support = 1,
        peri_s3_lower = 500, # was 350 mgHA/ccm
        peri_s3_upper = 10000, # very high value
        peri_s3_radius = 5,
        peri_s4_open_radius = 8,
        peri_s4_close_radius = 16
        ):
        """
        Initialization method.

        Parameters
        ----------
        in_value : int
            Integer value to use for voxels that are foreground.
            Default is 127.

        peri_s1_sigma : float
            Variance to use for the gaussian filtering in step 1 of the method
            that estimates the periosteal mask. Default is 1.5.

        peri_s1_support : int
            The support to use for the gaussian filtering in step 1 of the
            method that estimates the periosteal mask. Default is 1.

        peri_s1_lower : float
            Lower threshold for the threshold binarization in step 1 of the
            method that estimates the periosteal mask. Default is 400 HU.

        peri_s1_upper : float
            Upper threshold for the threshold binarization in step 1 of the
            method that estimates the periosteal mask. Default is 10000 HU.

        peri_s1_radius : int
            Radius of dilations in step 1 of the method that estimates the
            periosteal mask. Default is 35 voxels.

        peri_s2_sigma : float
            Variance to use for the gaussian filtering in step 2 of the method
            that estimates the periosteal mask. Default is 1.5.

        peri_s2_support : int
            The support to use for the gaussian filtering in step 2 of the
            method that estimates the periosteal mask. Default is 1.

        peri_s2_lower : float
            Lower threshold for the threshold binarization in step 2 of the
            method that estimates the periosteal mask. Default is 350 HU.

        peri_s2_upper : float
            Upper threshold for the threshold binarization in step 2 of the
            method that estimates the periosteal mask. Default is 10000 HU.

        peri_s2_radius : int
            Radius of dilations and erosions when performing the close
            with connected components labelling in step 2 of the method that
            estimates the periosteal mask. Default is 10 voxels.

        peri_s3_sigma : float
            Variance to use for the gaussian filtering in step 3 of the method
            that estimates the periosteal mask. Default is 1.5.

        peri_s3_support : int
            The support to use for the gaussian filtering in step 3 of the
            method that estimates the periosteal mask. Default is 1.

        peri_s3_lower : float
            Lower threshold for the threshold binarization in step 3 of the
            method that estimates the periosteal mask. Default is 500 HU.

        peri_s3_upper : float
            Upper threshold for the threshold binarization in step 3 of the
            method that estimates the periosteal mask. Default is 10000 HU.

        peri_s3_radius : int
            Radius of dilations and erosions when performing the close
            with connected components labelling in step 3 of the method that
            estimates the periosteal mask. Default is 5 voxels.

        peri_s4_open_radius : int
            Radius for the morphological opening in step 4 of the method that
            estimates the periosteal mask. Default is 8 voxels.

        peri_s4_close_radius : int
            Radius for the morphological closing in step 4 of the method that
            estimates the periosteal mask. Default is 16 voxels.

        """

        self.in_value = in_value
        self.out_value = 0 # this should only be zero

        self.peri_s1_sigma = peri_s1_sigma
        self.peri_s1_support = peri_s1_support
        self.peri_s1_lower = peri_s1_lower
        self.peri_s1_upper = peri_s1_upper
        self.peri_s1_radius = peri_s1_radius

        self.peri_s2_sigma = peri_s2_sigma
        self.peri_s2_support = peri_s2_support
        self.peri_s2_lower = peri_s2_lower
        self.peri_s2_upper = peri_s2_upper
        self.peri_s2_radius = peri_s2_radius

        self.peri_s3_sigma = peri_s3_sigma
        self.peri_s3_support = peri_s3_support
        self.peri_s3_lower = peri_s3_lower
        self.peri_s3_upper = peri_s3_upper
        self.peri_s3_radius = peri_s3_radius

        self.peri_s4_open_radius = peri_s4_open_radius
        self.peri_s4_close_radius = peri_s4_close_radius

        self.DEFAULT_MAX_ERROR = 0.01
        self.USE_SPACING = False

    def _gaussian_and_threshold(
        self, img,
        sigma, support,
        lower, upper
        ):
        """
        Gaussian smooth and then binarize an image using a threshold filter.

        Parameters
        ----------
        img : sitk.Image
            The input image.

        sigma : float
            Variance for the gaussian filtering.

        support : int
            Support for the gaussian filtering.

        lower: float
            Lower threshold for the binarization.

        upper : float
            Upper threshold for the binarization.

        Returns
        -------
        sitk.Image
            The binarized image.
        """

        # gaussian filtering
        img_gauss = sitk.DiscreteGaussian(img, sigma, support, self.DEFAULT_MAX_ERROR, self.USE_SPACING)

        # binary segmentation
        img_segmented = sitk.BinaryThreshold(img_gauss, lower, upper, self.in_value, self.out_value)

        return img_segmented

    def _get_largest_connected_component(self, img):
        """
        Get the largest connected component in a binary image.

        Parameters
        ----------
        img : sitk.Image
            The binary image to filter.

        Returns
        -------
        sitk.Image
            A binary image containing only the largest connected component from
            the input image.
        """

        img_conn = sitk.ConnectedComponent(img)
        img_conn = sitk.RelabelComponent(img_conn, sortByObjectSize=True)
        img_conn = self.in_value*(img_conn == 1)

        return img_conn

    def _invert_binary_image(self, img):
        """
        Swap the foreground and background of a binary image.

        Parameters
        ----------
        img : sitk.Image
            The binary image to invert.

        Returns
        -------
        sitk.Image
            The inverted binary image.
        """

        return self.in_value*(img!=self.in_value)

    def _close_with_connected_components(self, img, radius):
        """
        Perform a morphological closing operation on a binary image, except
        with a connected component filtering step to keep only the largest
        connected component of the background interposed between the dilation
        and the erosion.

        Parameters
        ----------
        img : sitk.Image
            The binary image to filter.

        radius : int
            The radius to use for the dilation and erosion.

        Returns
        -------
        sitk.Image
            The filtered image.
        """

        # dilate to close holes in cortex
        img = sitk.BinaryDilate(
            img, [radius]*3, sitk.sitkBall,
            self.out_value, self.in_value
        )

        # invert the image to switch to background
        img = self._invert_binary_image(img)

        # perform connected components on background
        img = self._get_largest_connected_component(img)

        # reinvert to get back to foreground
        img = self._invert_binary_image(img)

        # erode back the dilated bone volume
        img = sitk.BinaryErode(
            img, [radius]*3, sitk.sitkBall,
            self.out_value, self.in_value
        )

        return img

    def get_periosteal_mask(self, img):
        """
        Compute the periosteal mask from an input image.

        Parameters
        ----------
        img : sitk.Image
            The gray-scale AIM. Currently this is written for images in HU,
            if you want to input a density image then you'll need to modify
            the lower and upper thresholds to be in the correct units.

        Returns
        -------
        sitk.Image
            A binary image that is the periosteal mask.
        """

        # STEP 1: Mask out the largest bone only

        img_segmented = self._gaussian_and_threshold(
            img, self.peri_s1_sigma, self.peri_s1_support,
            self.peri_s1_lower, self.peri_s1_upper
        )

        # component labelling to keep only largest region
        img_segmented = self._get_largest_connected_component(img_segmented)

        # dilation
        # !!!NOTE: I'm using a Euclidean metric for the structirng element,
        # the IPL implementation uses the 3-4-5 chamfer metric. Feel free to
        # swap out the code if you can figure out how to get the 3-4-5
        # chamfer metric in SimpleITK

        img_segmented = sitk.BinaryDilate(
            img_segmented,
            [self.peri_s1_radius]*3, sitk.sitkBall,
            self.out_value, self.in_value
        )

        # invert the image to get the background
        img_segmented = self._invert_binary_image(img_segmented)

        # connected components on the background
        img_segmented = self._get_largest_connected_component(img_segmented)

        # invert back to foreground
        img_segmented = self._invert_binary_image(img_segmented)

        # mask the original image using this segmentation
        img_masked = sitk.Mask(img, img_segmented)

        # and save the segmentation to use later
        img_segmented_s1 = img_segmented

        # STEP 2: create a mask with a low threshold

        # threshold using low threshold
        img_segmented = self._gaussian_and_threshold(
            img_masked, self.peri_s2_sigma, self.peri_s2_support,
            self.peri_s2_lower, self.peri_s2_upper
        )

        # keep only largest component
        img_segmented = self._get_largest_connected_component(img_segmented)

        # dilate/conn comp/erode to close holes in cortex
        img_segmented = self._close_with_connected_components(
            img_segmented, self.peri_s2_radius
        )

        # now mask the image using the latest segmentation
        img_masked = sitk.Mask(img, img_segmented)

        # save the final segmentation from step 2
        img_segmented_s2 = img_segmented

        # STEP 3: create another mask with a slightly higher threshold

        # gaussian blur and segment with higher threshold
        img_segmented = self._gaussian_and_threshold(
            img_masked, self.peri_s3_sigma, self.peri_s3_support,
            self.peri_s3_lower, self.peri_s3_upper
        )

        # dilate/conn comp/erode to close holes in cortex
        img_segmented = self._close_with_connected_components(
            img_segmented, self.peri_s3_radius
        )

        # again, mask the image with the new segmentation
        img_masked = sitk.Mask(img, img_segmented)

        # save the final segmentation from step 3
        img_segmented_s3 = img_segmented


        # STEP 4: Create the final segmentation using the segmentations from
        # steps 2 and 3

        # find where the two masks are different
        img_segmented_diff = self.in_value*((img_segmented_s2==self.in_value)!=(img_segmented_s3==self.in_value))

        # do an opening on the diff
        img_segmented_diff_open = sitk.BinaryMorphologicalOpening(
            img_segmented_diff, [self.peri_s4_open_radius]*3, sitk.sitkBall,
            self.out_value, self.in_value
        )

        # combine this with the segmentation from step 3
        peri_mask = self.in_value*sitk.Or(
            img_segmented_s3 == self.in_value,
            img_segmented_diff_open == self.in_value
        )

        # perform a smoothening close
        # !!NOTE: This operation is in the original script but it seems to do
        # more harm than good on the knee scans: the bone surface is not convex
        # so ending with a high-radius close smooths the mask over concave
        # surface features. Qualitatively, it seems to me like a candidate for
        # improving the algorithm would be to replace this with an opening
        peri_mask = sitk.BinaryMorphologicalClosing(
            peri_mask, [self.peri_s4_close_radius]*3, sitk.sitkBall,
            self.in_value
        )

        # mask the final peri mask using the first rough mask we created in
        # step 1
        peri_mask = sitk.Mask(peri_mask, img_segmented_s1)

        return peri_mask


    def get_endosteal_mask(self, img, peri):
        """
        Compute the endosteal mask from the image and periosteal mask.

        Parameters
        ----------
        img : sitk.Image
            The gray-scale AIM. Currently this is written for images in HU,
            if you want to input a density image then you'll need to modify
            the lower and upper thresholds to be in the correct units.

        peri : sitk.Image
            A binary image that should be the periosteal mask.

        Returns
        -------
        sitk.Image
            A binary image that is the endosteal mask.
        """
        pass

    def get_masks(self, img):
        """
        Combined method to compute both the periosteal and endosteal masks.

        Parameters
        ----------
        img : sitk.Image
            The gray-scale AIM. Currently this is written for images in HU,
            if you want to input a density image then you'll need to modify
            the lower and upper thresholds to be in the correct units.

        Returns
        -------
        (sitk.Image, sitk.Image)
            Tuple of two binary images. The first image is the periosteal mask
            and the second image is the endosteal mask.
        """
        pass

    def __str__(self):
        return f'Autocontour object (--str to be implemented--).'

    def __repr__(self):
        return 'Autocontour(--repr to be implemented--)'
