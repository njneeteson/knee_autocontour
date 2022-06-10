import SimpleITK as sitk

class AutocontourKnee:
    """
    A class for computing the periosteal and endosteal masks for a knee
    HR-pQCT image using only thresholding and morphological operations.

    Attributes
    ----------
    in_value : int

    sigma0 : float

    support0 : int

    lower0 : float

    upper0 : float

    sigma1 : float

    support1 : int

    lower1 : float

    upper1 : float

    misc1 : int

    sigma2 : float

    support2 : int

    lower2 : float

    upper2 : float

    misc1_2 : int

    misc1_3 : int

    misc1_4 : int


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
        sigma0 = 1.5,
        support0 = 1,
        lower0 = 400, # was 350 mgHA/ccm
        upper0 = 10000, # very high value
        sigma1 = 1.5,
        support1 = 1,
        lower1 = 350, # was 250 mgHA/ccm
        upper1 = 10000,
        misc1 = 10,
        sigma2 = 1.5,
        support2 = 1,
        lower2 = 500, # was 350 mgHA/ccm
        upper2 = 10000, # very high value
        misc1_2 = 5,
        misc1_3 = 8,
        misc1_4 = 16
        ):
        """
        Initialization method.

        Parameters
        ----------
        in_value : int
            Integer value to use for voxels that are foreground.
            Default is 127.

        sigma0 : float
            Variance to use for the gaussian filtering in step 1 of the method
            that estimates the periosteal mask. Default is 1.5.

        support0 : int
            The support to use for the gaussian filtering in step 1 of the
            method that estimates the periosteal mask. Default is 1.

        lower0 : float
            Lower threshold for the threshold binarization in step 1 of the
            method that estimates the periosteal mask. Default is 400 HU.

        upper0 : float
            Upper threshold for the threshold binarization in step 1 of the
            method that estimates the periosteal mask. Default is 10000 HU.

        sigma1 : float
            Variance to use for the gaussian filtering in step 2 of the method
            that estimates the periosteal mask. Default is 1.5.

        support1 : int
            The support to use for the gaussian filtering in step 2 of the
            method that estimates the periosteal mask. Default is 1.

        lower1 : float
            Lower threshold for the threshold binarization in step 2 of the
            method that estimates the periosteal mask. Default is 350 HU.

        upper1 : float
            Upper threshold for the threshold binarization in step 2 of the
            method that estimates the periosteal mask. Default is 10000 HU.

        misc1 : int
            Number of dilations and erosions to use when performing the close
            with connected components labelling in step 2 of the method that
            estimates the periosteal mask. Default is 10 voxels.

        sigma2 : float
            Variance to use for the gaussian filtering in step 3 of the method
            that estimates the periosteal mask. Default is 1.5.

        support2 : int
            The support to use for the gaussian filtering in step 3 of the
            method that estimates the periosteal mask. Default is 1.

        lower2 : float
            Lower threshold for the threshold binarization in step 2 of the
            method that estimates the periosteal mask. Default is 350 HU.

        upper2 : float
            Upper threshold for the threshold binarization in step 2 of the
            method that estimates the periosteal mask. Default is 10000 HU.

        misc1_2 : int

        misc1_3 : int

        misc1_4 : int

        """

        self.in_value = in_value
        self.out_value = 0 # don't

        self.sigma0 = sigma0
        self.support0 = support0
        self.lower0 = lower0
        self.upper0 = upper0

        self.sigma1 = sigma1
        self.support1 = support1
        self.lower1 = lower1
        self.upper1 = upper1
        self.misc1 = misc1

        self.sigma2 = sigma2
        self.support2 = support2
        self.lower2 = lower2
        self.upper2 = upper2
        self.misc1_2 = misc1_2

        self.misc1_3 = misc1_3
        self.misc1_4 = misc1_4

        self.DEFAULT_MAX_ERROR = 0.01
        self.USE_SPACING = False

    def _gaussian_and_threshold(
        self, img,
        sigma, support,
        lower, upper
        ):

        # gaussian filtering
        img_gauss = sitk.DiscreteGaussian(img, sigma, support, self.DEFAULT_MAX_ERROR, self.USE_SPACING)

        # binary segmentation
        img_segmented = sitk.BinaryThreshold(img_gauss, lower, upper, self.in_value, self.out_value)

        return img_segmented

    def _get_largest_connected_component(self, img):

        img_conn = sitk.ConnectedComponent(img)
        img_conn = sitk.RelabelComponent(img_conn, sortByObjectSize=True)
        img_conn = self.in_value*(img_conn == 1)

        return img_conn

    def _invert_binary_image(self, img):

        return self.in_value*(img!=self.in_value)

    def _close_with_connected_components(self, img, radius):

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

        # STEP 1: Mask out the largest bone only

        img_segmented = self._gaussian_and_threshold(
            img, self.sigma0, self.support0,
            self.lower0, self.upper0
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
            [35, 35, 35], sitk.sitkBall,
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
            img_masked, self.sigma1, self.support1,
            self.lower1, self.upper1
        )

        # keep only largest component
        img_segmented = self._get_largest_connected_component(img_segmented)

        # dilate/conn comp/erode to close holes in cortex
        img_segmented = self._close_with_connected_components(
            img_segmented, self.misc1
        )

        # now mask the image using the latest segmentation
        img_masked = sitk.Mask(img, img_segmented)

        # save the final segmentation from step 2
        img_segmented_s2 = img_segmented

        # STEP 3: create another mask with a slightly higher threshold

        # gaussian blur and segment with higher threshold
        img_segmented = self._gaussian_and_threshold(
            img_masked, self.sigma2, self.support2,
            self.lower2, self.upper2
        )

        # dilate/conn comp/erode to close holes in cortex
        img_segmented = self._close_with_connected_components(
            img_segmented, self.misc1_2
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
            img_segmented_diff, [self.misc1_3]*3, sitk.sitkBall,
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
            peri_mask, [self.misc1_4]*3, sitk.sitkBall,
            self.in_value
        )

        # mask the final peri mask using the first rough mask we created in
        # step 1
        peri_mask = sitk.Mask(peri_mask, img_segmented_s1)

        return peri_mask


    def get_endosteal_mask(self, img, peri):
        pass

    def get_masks(self, img):
        pass

    def __str__(self):
        return f'Autocontour object (--str to be implemented--).'

    def __repr__(self):
        return 'Autocontour(--repr to be implemented--)'
