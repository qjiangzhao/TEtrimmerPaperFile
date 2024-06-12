import os
import numpy as np
import matplotlib.pyplot as plt
from Bio import AlignIO
from Bio.Seq import Seq
import pandas as pd
import seaborn as sns
from scipy import stats
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment

# To run this script, please install pandas, biopython, and numpy


#####################################################################################################
# Code block: Define MSA cleaning class
#####################################################################################################
class CropEnd:
    """
    Crop each single sequence end of the MSA by nucleotide divergence.
    """

    def __init__(self, alignment, threshold=16, window_size=20, save_cropped_msa=False, output_path=None, input_n=None):
        """
        :param input_file: str, path to the multiple sequence alignment
        :param threshold: default 16, nucleotide number inside the checking window whose proportion should be greater than 80%
        :param window_size: default 20, checking window size to define start and end position
        """

        self.alignment = alignment
        self.threshold = threshold
        self.window_size = window_size
        self.threshold_converted = round(self.threshold/self.window_size, 2)
        self.save_cropped_msa = save_cropped_msa
        self.output_path = output_path
        self.input_n = input_n
        # Define a dictionary the key are sequence names, the values are a list contains nucleotides proportion
        self.proportions_dict = {record.id: [] for record in self.alignment}
        # Define a dictionary to hold start and end positions
        self.position_dict = {record.id: [0, 0] for record in self.alignment}
        # Define an empty dataframe to store proportion information
        self.df = None
        self.cropped_alignment = []
        self.pro_calculation()
        self.find_positions()
        self.crop_alignment()
        if save_cropped_msa and output_path is not None:
            self.write_to_file(os.path.join(self.output_path, self.input_n))

    def pro_calculation(self):
        """
        :function pro_calculation: calculate nucleotide proportion in each column of the MSA
        :return: a data frame containing all sequence names and the respective nucleotide proportion information for each position
        """
        # Loop through each column of the alignment
        for i in range(self.alignment.get_alignment_length()):
            # Count the number of each nucleotide in this column
            counts = {"a": 0, "c": 0, "g": 0, "t": 0}
            for record in self.alignment:
                nucleotide = record.seq[i].lower()
                if nucleotide in counts:
                    counts[nucleotide] += 1

            # Calculate the proportion of each nucleotide
            total = sum(counts.values())

            # Generate a dictionary named proportions contains nucleotide proportion for this column
            if total < 5:  # Ignore columns less 5 nucleotides
                proportions = {nucleotide: 0 for nucleotide in counts}
            else:
                proportions = {nucleotide: count / total for nucleotide, count in counts.items()}

            # Add the proportion of the nucleotide at this position to each sequence
            for record in self.alignment:  # This will loop each sequences in alignment
                nucleotide = record.seq[i]  # Refer to that column
                if nucleotide in proportions:
                    # Write proportion information into proportions_dict
                    self.proportions_dict[record.id].append(proportions[nucleotide])
                else:
                    # When there is a gap, use number 0 replace proportion
                    self.proportions_dict[record.id].append(np.nan)

        # Convert the dictionary to a DataFrame
        self.df = pd.DataFrame(self.proportions_dict)
        self.df = self.df.transpose()  # transpose the DataFrame so that each row represents a sequence
        self.df.columns = range(1,
                                self.alignment.get_alignment_length() + 1)  # rename the columns to represent positions
        # Convert to two decimal numbers
        self.df = self.df.round(2)

    def find_positions(self):
        """
        This function will define the start and end position for each sequence
            by nucleotide proportions.

        :return: a dictionary that contains sequence name, start, and end positions
        """
        # Loop over the DataFrame's rows
        for index, row in self.df.iterrows():
            # Find start position
            for i in range(len(row) - self.window_size + 1):
                window = row[i:i + self.window_size]
                if window.sum() > self.threshold:
                    self.position_dict[index][0] = i
                    break
            # Find end position
            for i in range(len(row) - 1, self.window_size - 2, -1):
                window = row[i - self.window_size + 1:i + 1]
                if window.sum() > self.threshold:
                    self.position_dict[index][1] = i + 1  # add 1 to make the position 1-indexed
                    break

    def crop_alignment(self):
        # Create a new list to hold the cropped sequences
        # Loop through each sequence in the alignment
        for record in self.alignment:
            # Create a new string with the cropped sequence
            cropped_seq = "-" * self.position_dict[record.id][0] + \
                          str(record.seq[self.position_dict[record.id][0]:self.position_dict[record.id][1]]) + \
                          "-" * (len(record.seq) - self.position_dict[record.id][1])
            # Create a new SeqRecord with the cropped sequence and add it to the list
            self.cropped_alignment.append(SeqRecord(Seq(cropped_seq), id=record.id, description=""))
        # Convert the list of SeqRecords into a MultipleSeqAlignment
        self.cropped_alignment = MultipleSeqAlignment(self.cropped_alignment)

        return self.cropped_alignment

    def average_proportion_per_column(self):
        """
        Calculate the average proportion for each column across all sequences.
        """
        # Ensure the DataFrame is available
        if not hasattr(self, 'df'):
            raise ValueError("The DataFrame has not been created yet. Please run pro_calculation() first.")

        # Calculate the average for each column
        average_proportions = self.df.mean()

        # Calculate the overall mean of the column averages
        overall_average = average_proportions.mean()

        return average_proportions, overall_average

    def write_to_file(self, output_file):
        output_file = f"{output_file}_{self.threshold_converted}_TEtrimmer_clean.fa"
        with open(output_file, "w") as f:
            AlignIO.write(self.cropped_alignment, f, "fasta")
        return output_file


def count_nucleotides(alignment):
    """Count total nucleotides (non-gap characters) in an alignment."""
    total_nucleotides = 0
    for record in alignment:
        total_nucleotides += sum(1 for nuc in record.seq if nuc in ['A', 'G', 'C', 'T', 'a', 'g', 'c', 't'])
    return total_nucleotides


def count_gap_differences(treated1, treated2):
    """Count differences in gap placements between two alignments."""
    gap_in_manual_not_software = 0
    gap_in_software_not_manual = 0

    for seq_index in range(len(treated1)):  # Iterate over each sequence
        for pos_index in range(len(treated1[seq_index])):  # Iterate over each position
            nuc1 = treated1[seq_index][pos_index]
            nuc2 = treated2[seq_index][pos_index]
            if nuc1 == "-" and nuc2 != "-":
                gap_in_manual_not_software += 1
            elif nuc1 != "-" and nuc2 == "-":
                gap_in_software_not_manual += 1

    return gap_in_manual_not_software, gap_in_software_not_manual


def load_alignment(file_path):
    """
    Load an alignment from a FASTA file and sort it by sequence names (IDs).

    :param file_path: Path to the FASTA file containing the multiple sequence alignment.
    :return: A MultipleSeqAlignment object with sequences sorted by their IDs.
    """
    # Load the alignment
    alignment = AlignIO.read(file_path, "fasta")

    # Convert the alignment to a list of SeqRecord objects
    alignment_list = list(alignment)

    # Sort the list by sequence IDs
    sorted_alignment_list = sorted(alignment_list, key=lambda x: x.id)

    # Create a new MultipleSeqAlignment object from the sorted list
    sorted_alignment = MultipleSeqAlignment(sorted_alignment_list)

    return sorted_alignment


#####################################################################################################
# Code block: Define confusion matrix benchmarking function to evaluate MSA cleaning
#####################################################################################################
def calculate_clean_efficiency(original_align_path, manual_align_path, thr=0.8, window=20, output_path=None):

    # Load treated alignments
    original_alignment = load_alignment(original_align_path)
    manual_alignment = load_alignment(manual_align_path)

    # Crop alignment
    cropend_thr = round(thr * window)

    # Use this line of code to save cropped MSA to a file
    clean_obj = CropEnd(original_alignment, cropend_thr, window, save_cropped_msa=True, output_path=output_path,
                        input_n=os.path.basename(original_align_path))
    cleanned_align = clean_obj.crop_alignment()

    # Calculate manual cleaned alignment nucleotide number
    manual_nucl_n = count_nucleotides(manual_alignment)

    # Count the gap differences
    gap_in_manual_not_original, gap_in_original_not_manual = count_gap_differences(manual_alignment, original_alignment)
    gap_in_manual_not_software, gap_in_software_not_manual = count_gap_differences(manual_alignment,
                                                                                   cleanned_align)

    # clean_efficiency = (gap_in_manual_not_original - gap_in_manual_not_software) / gap_in_manual_not_original
    # false_rate = gap_in_software_not_manual / manual_nucl_n
    # gap_in_manual_not_software: FN (false negative)
    # gap_in_software_not_manual: FP (false positive)
    # gap_in_manual_not_original - gap_in_manual_not_software: TP (true positive)
    # gap_in_manual_not_original: TP + FN (nucleotide need to be removed)
    # manual_nucl_n: FP + TN (nucleotide need to be kept)
    # manual_nucl_n - gap_in_software_not_manual: TN (True negative)

    # Calculate TP, FN, FP, TN
    TP = gap_in_manual_not_original - gap_in_manual_not_software
    FN = gap_in_manual_not_software
    FP = gap_in_software_not_manual
    TN = manual_nucl_n - gap_in_software_not_manual

    # Calculate performance metrics
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

    # F1 Score
    F1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    # False Discovery Rate (FDR)
    FDR = FP / (FP + TP) if (FP + TP) > 0 else 0

    # Return all metrics
    return sensitivity, specificity, accuracy, precision, F1, FDR


#####################################################################################################
# Code block: Perform MSA cleaning and benchmarking
#####################################################################################################

os.chdir("Path_to_folder_contains_all_original_and_manually_cleaned_MSA")

alignment_pairs = [
    ("rnd_4_family_831#LTR__Gypsy.anno.fa", "rnd_4_family_831#LTR__Gypsy.anno_manual_clean.fa"),
    ("rnd_4_family_83#LTR__Copia.anno.fa", "rnd_4_family_83#LTR__Copia.anno_manual_clean.fa"),
    ("rnd_1_family_79#LINE__I-Jockey.anno.fa", "rnd_1_family_79#LINE__I-Jockey.anno_manual_clean.fa"),
    ("rnd_1_family_380#LINE__I.anno.fa", "rnd_1_family_380#LINE__I.anno_manual_clean.fa"),
    ("rnd_1_family_362#LTR__Gypsy.anno.fa", "rnd_1_family_362#LTR__Gypsy.anno_manual_clean.fa"),
    ("ltr_1_family_7#LTR__Gypsy.anno.fa", "ltr_1_family_7#LTR__Gypsy.anno_manual_clean.fa"),
    ("ltr_1_family_66#LINE__I-Jockey.anno.fa", "ltr_1_family_66#LINE__I-Jockey.anno_manual_clean.fa"),
    ("ltr_1_family_202#LTR__Gypsy.anno.fa", "ltr_1_family_202#LTR__Gypsy.anno_manual_clean.fa"),
    ("ltr_1_family_302#DNA__hat-hat5.anno.fa","ltr_1_family_302#DNA__hat-hat5.anno_manual_clean.fa"),
    ("ltr_1_family_328#DNA__TcMar-Tc1.anno.fa","ltr_1_family_328#DNA__TcMar-Tc1.anno_manual_clean.fa")
]

output_path = "Define_outpu_path"

# Preparation for plotting individual points with error bars
thresholds = np.linspace(0, 1, 21)
all_sensitivity = {thr: [] for thr in thresholds}  # Dictionary to store all efficiencies for each threshold
all_specificity = {thr: [] for thr in thresholds}
all_accuracy = {thr: [] for thr in thresholds}         # Dictionary to store all rates for each threshold
all_precision = {thr: [] for thr in thresholds}
all_f1 = {thr: [] for thr in thresholds}
all_fdr = {thr: [] for thr in thresholds}

# Loop to fill the dictionaries with individual clean efficiencies and false rates
for thr in thresholds:
    for original, manual in alignment_pairs:
        sensitivity, specificity, accuracy, precision, F1, FDR = calculate_clean_efficiency(original, manual, thr, 40, output_path=output_path)
        all_sensitivity[thr].append(sensitivity)
        all_specificity[thr].append(specificity)
        all_accuracy[thr].append(accuracy)
        all_precision[thr].append(precision)
        all_f1[thr].append(F1)
        all_fdr[thr].append(FDR)


def ave_mean_std_err_cal(dict, thresholds):
    ave_n = [np.mean(dict[thr]) for thr in thresholds]
    std_e = [stats.sem(dict[thr]) if dict[thr] else 0 for thr in thresholds]
    return ave_n, std_e


ave_sensitivity, std_error_sensitivity = ave_mean_std_err_cal(all_sensitivity, thresholds)
ave_specificity, std_error_specificity = ave_mean_std_err_cal(all_specificity, thresholds)
ave_accuracy, std_error_accuracy = ave_mean_std_err_cal(all_accuracy, thresholds)
ave_precision, std_error_precision = ave_mean_std_err_cal(all_precision, thresholds)
ave_f1, std_error_f1 = ave_mean_std_err_cal(all_f1, thresholds)
ave_fdr, std_error_fdr = ave_mean_std_err_cal(all_fdr, thresholds)


# Store to file
# Prepare the data
data = {
    "Threshold": thresholds,
    "AverageSensitivity": ave_sensitivity,
    "StdErrorSensitivity": std_error_sensitivity,
    "AverageSpecificity": ave_specificity,
    "StdErrorSpecificity": std_error_specificity,
    "AverageAccuracy": ave_accuracy,
    "StdErrorAccuracy": std_error_accuracy,
    "AveragePrecision": ave_precision,
    "StdErrorPrecision": std_error_precision,
    "AverageF1": ave_f1,
    "StdErrorF1": std_error_f1,
    "AverageFDR": ave_fdr,
    "StdErrorFDR": std_error_f1,
}

# Convert the data into a pandas DataFrame
results_df = pd.DataFrame(data)

# Specify the file path where you want to save the CSV
file_path = 'clean_efficiency_and_false_rate_metrics.csv'

# Save the DataFrame to a CSV file
results_df.to_csv(file_path, index=False)

# Replace 'file_path' with the actual path where the CSV file is saved
file_path = 'clean_efficiency_and_false_rate_metrics.csv'

# Read the DataFrame from the CSV file
results_df = pd.read_csv(file_path)

# Extracting data for plotting
thresholds = results_df['Threshold']
avg_sensitivity = results_df['AverageSensitivity']
std_error_sensitivity = results_df['StdErrorSensitivity']

avg_accuracy = results_df['AverageAccuracy']
std_error_accuracy = results_df['StdErrorAccuracy']

avg_precision = results_df['AveragePrecision']
std_error_precision = results_df['StdErrorPrecision']

avg_f1 = results_df['AverageF1']
std_error_f1 = results_df['StdErrorF1']

#####################################################################################################
# Code block: Plot benchmarking result
#####################################################################################################

# Setting the seaborn style to "ticks" for minor tick marks
sns.set_style("ticks")
fig, ax1 = plt.subplots(figsize=(10, 6))

# Custom X-axis ticks and labels
step_size = 2  # Adjust this based on your preference and the total number of ticks

# Use slicing with step_size to select specific ticks and labels
xticks = thresholds[::step_size]  # Select every nth tick according to step_size
ax1.set_xticks(xticks)
xticklabels = ['{:.1f}'.format(tick) for tick in xticks]  # Format labels to two decimal places
ax1.set_xticklabels(xticklabels)  # Set custom labels with optional rotation for better readability

# Set x axis label
ax1.set_xlabel('Cleaning threshold', fontsize=20, weight='bold', labelpad=10)
ax1.tick_params(axis='x', colors='black', which='both', labelsize=14)

# Setting the Y-axis labels and tick parameters to red
ax1.tick_params(axis='y', colors='tab:red', which='both', labelsize=14)  # Ticks

# Plotting True cleaning rate
ax1.errorbar(thresholds, avg_sensitivity, yerr=std_error_sensitivity, fmt='-o', color='tab:green', linewidth=2)
ax1.set_ylabel('Cleaning sensitivity', color='tab:green', fontsize=18, weight='bold')

# Plotting precision
ax1.errorbar(thresholds, avg_precision, yerr=std_error_precision, fmt='-o', color='tab:blue', linewidth=2)

# Setting up the secondary Y-axis for Average False Rate
ax2 = ax1.twinx()

ax2.tick_params(axis='y', colors='tab:blue', which='both', labelsize=14)  # Ticks

# Plotting False cleaning rate
ax2.errorbar(thresholds, avg_f1, yerr=std_error_f1, fmt='-s', color='tab:orange', linewidth=2)
ax2.set_ylabel('Cleaning F1 score', color='tab:orange', fontsize=18, weight='bold')

# Disabling the grid to avoid visual clutter
ax1.grid(False)
ax2.grid(False)

fig.tight_layout()
plt.show()


"""

os.chdir("/Users/panstrugamacbook/Documents/PhD_project_files/Papers/TE_trimmer/Pictures/MSA_cleaning/rnd_1_family_34.fasta_cleaning_at_different_divergence")

alignment_pairs = [
    ("/Users/panstrugamacbook/Documents/PhD_project_files/Papers/TE_trimmer/Pictures/MSA_cleaning/rnd_1_family_34.fasta.b.bed_uf.bed_g_1.bed_fm.bed_0_0_bcln.fa_aln.fa_cl.fa_gs_after_cleaning.fa",
     "/Users/panstrugamacbook/Documents/PhD_project_files/Papers/TE_trimmer/Pictures/MSA_cleaning/rnd_1_family_34.fasta.b.bed_uf.bed_g_1.bed_fm.bed_0_0_bcln.fa_aln.fa_cl.fa_manually_cleaned.fa")
]

alignment_pairs = [
    ("rnd_4_family_831#LTR__Gypsy.anno.fa", "rnd_4_family_831#LTR__Gypsy.anno.clean.fa"),
    ("rnd_4_family_83#LTR__Copia.anno.fa", "rnd_4_family_83#LTR__Copia.anno.clean.fa"),
    ("rnd_1_family_79#LINE__I-Jockey.anno.fa", "rnd_1_family_79#LINE__I-Jockey.anno.clean.fa"),
    ("rnd_1_family_380#LINE__I.anno.fa", "rnd_1_family_380#LINE__I.anno.clean.fa"),
    ("rnd_1_family_362#LTR__Gypsy.anno.fa", "rnd_1_family_362#LTR__Gypsy.anno.clean.fa"),
    ("ltr_1_family_7#LTR__Gypsy.anno.fa", "ltr_1_family_7#LTR__Gypsy.anno._clean.fa"),
    ("ltr_1_family_66#LINE__I-Jockey.anno.fa", "ltr_1_family_66#LINE__I-Jockey.anno.clean.fa"),
    ("ltr_1_family_202#LTR__Gypsy.anno.fa", "ltr_1_family_202#LTR__Gypsy.anno.clean.fa"),
    ("ltr_1_family_302#DNA__hat-hat5.anno.fa","ltr_1_family_302#DNA__hat-hat5.anno.clean.fa"),
    ("ltr_1_family_328#DNA__TcMar-Tc1.anno.fa","ltr_1_family_328#DNA__TcMar-Tc1.anno.clean.fa")
]
"""