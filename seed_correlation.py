from nilearn import datasets
from nilearn import input_data
import numpy as np
from nilearn import plotting

adhd_dataset = datasets.fetch_adhd(n_subjects=1)
func_filename = adhd_dataset.func[0]
confound_filename = adhd_dataset.confounds[0]

print(func_filename)
print(confound_filename)
pcc_coords = [(0, -52, 18)]

seed_masker = input_data.NiftiSpheresMasker(
    pcc_coords, radius=8,
    detrend=True, standardize=True,
    low_pass=0.1, high_pass=0.01, t_r=2.,
    memory='nilearn_cache', memory_level=1, verbose=0)


seed_time_series = seed_masker.fit_transform(func_filename,
                                             confounds=[confound_filename])

brain_masker = input_data.NiftiMasker(
    smoothing_fwhm=6,
    detrend=True, standardize=True,
    low_pass=0.1, high_pass=0.01, t_r=2.,
    memory='nilearn_cache', memory_level=1, verbose=0)

brain_time_series = brain_masker.fit_transform(func_filename,
                                               confounds=[confound_filename])

print("seed time series shape: (%s, %s)" % seed_time_series.shape)
print("brain time series shape: (%s, %s)" % brain_time_series.shape)



seed_based_correlations = np.dot(brain_time_series.T, seed_time_series) / \
                          seed_time_series.shape[0]

print("seed-based correlation shape: (%s, %s)" % seed_based_correlations.shape)
print("seed-based correlation: min = %.3f; max = %.3f" % (
    seed_based_correlations.min(), seed_based_correlations.max()))

seed_based_correlations_fisher_z = np.arctanh(seed_based_correlations)
print("seed-based correlation Fisher-z transformed: min = %.3f; max = %.3f" % (
    seed_based_correlations_fisher_z.min(),
    seed_based_correlations_fisher_z.max()))

# Finally, we can tranform the correlation array back to a Nifti image
# object, that we can save.
seed_based_correlation_img = brain_masker.inverse_transform(
    seed_based_correlations.T)
seed_based_correlation_img.to_filename('sbc_z.nii.gz')


display = plotting.plot_stat_map(seed_based_correlation_img, threshold=0.3,
                                 cut_coords=pcc_coords[0])
display.add_markers(marker_coords=pcc_coords, marker_color='g',
                    marker_size=300)
# At last, we save the plot as pdf.
display.savefig('sbc_z.pdf')


