# Genomics and Clinical Data Analysis (2023 Spring)

## Practice Session #1: Sequencing

In this session, we will learn how to convert raw unmapped read files (`FASTQ`) to analysis-ready files (`VCF`). \
The overall process in this session is based on the [GATK Best Practice](https://gatk.broadinstitute.org/hc/en-us/categories/360002302312-Getting-Started). \
This document was created on March 14, 2023 and the following contents were tested on local WSL (Ubuntu 22.04.1 LTS).
### 0. Installing Linux and Anaconda in Windows
Using Linux has become easy in Windows with WSL. \
To start, launch windows powershell in administration mode and run following. 
``` 
wsl --install
```
After system restart, linux can be run from terminal app. (Note that hard drive is mounted under /mnt)

```

# find latest release for Linux-x86 and copy link from https://www.anaconda.com/products/distribution
wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh

# install anaconda by running bash and follow instructions
bash Anaconda3-2022.10-Linux-x86_64.sh

```
### 1. Setting up the environment

We will use the Anaconda environment on the GSDS cluster. \
It is already created on the GSDS cluster, but you can create the environment on your local machine with the following command \
In this session, OpenJDK, samtools, GATK and BWA are installed in creation of conda environment and Picard is downloaded as java package

```
# Create conda environment and install softwares 
conda create -n SEQ samtools gatk4 bwa -c anaconda -c bioconda
conda activate SEQ

# Install jdk 17 version (Required after picard 3.0.0)
wget https://download.java.net/java/GA/jdk17.0.2/dfd4a8d0985749f896bed50d7138ee7f/8/GPL/openjdk-17.0.2_linux-x64_bin.tar.gz
tar xvf openjdk-17.0.2_linux-x64_bin.tar.gz
export JAVA_HOME=[PATH_TO_JDK]/jdk-17.0.2/
export HOME=$HOME:$JAVA_HOME/bin

# Download Picard (Find Latest Release: https://github.com/broadinstitute/picard/releases/latest)
wget https://github.com/broadinstitute/picard/releases/download/3.0.0/picard.jar

```

### 2. Preparing data

To map our raw unmapped reads, we need the reference panel and the information for known variants. \
Here, we will use the `FASTA` file of 1000 Genome Phase 3 (GRCh37 build) and the `VCF` file for known variants. \
You can browse FTP server (ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/) of 1000 Genome Project.

```
# Download 1000 Genome reference panel
wget http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/human_g1k_v37.fasta.gz
gzip -d human_g1k_v37.fasta.gz

# Download VCF file and its index (tbi) file for known variants
wget http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/integrated_sv_map/ALL.wgs.mergedSV.v8.20130502.svs.genotypes.vcf.gz
wget http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/integrated_sv_map/ALL.wgs.mergedSV.v8.20130502.svs.genotypes.vcf.gz.tbi
```

You can check the contents of the `FASTA` file by the following command:

```
# View first 200 rows of FASTA file
head -200 human_g1k_v37.fasta
```

We need to preprocess (create index/dict files) the reference genome files.

```
# Create index (.fai)
samtools faidx human_g1k_v37.fasta

# Create dict (.dict)
gatk CreateSequenceDictionary -R human_g1k_v37.fasta

# Construct files with Burrows-Wheeler Transformation (5 files)
bwa index -a bwtsw human_g1k_v37.fasta
```

And we need a sequence read file (`FASTQ`) for the sample individual (HG00096).

```
# Download sequence read file
# sample HG00096
wget http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data/HG00096/sequence_read/SRR062634.filt.fastq.gz
gzip -d SRR062634.filt.fastq.gz
# sample HG00097
wget http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data/HG00097/sequence_read/SRR741384.filt.fastq.gz
gzip -d SRR741384.filt.fastq.gz
# sample HG00099
wget http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data/HG00099/sequence_read/SRR741411.filt.fastq.gz
gzip -d SRR741411.filt.fastq.gz
```

We can check the contents of the `FASTQ` file by `head` command, and you will see the following:

```
@SRR062634.321 HWI-EAS110_103327062:6:1:1446:951/2
TGATCATTTGATTAATACTGACATGTAGACAAGAAGAAAAGTATGTTTCATGCTATTTTGAGTAACTTCCATTTAGAAGCCTACTCCTGAGCACAACATT
+
B5=BD5DAD?:CBDD-DDDDDCDDB+-B:;?A?CCE?;D3A?B?DB??;DDDEEABD+>DAC?A-CD-=D?C5A@::AC-?AB?=:>CA@##########
@SRR062634.488 HWI-EAS110_103327062:6:1:1503:935/2
AATGTTATTAAAAATGGACACCTTTTTCTCACACATTCAGTTTCATTGTCTCGCACCCCATCGTTTTACTTTTCTTCCTTCAGAAAATGATAAATGTGGG
+
AAAA?5D?BD==ADBD:DBDDDDD5D=;@>AD-CD?D=C5=@4<7CCAA5?=?>5@BC?*<:=>>:D:B5?B?5?'3::5?5<:;*97:<A#########
@SRR062634.849 HWI-EAS110_103327062:6:1:1587:921/2
CAGATCAGAATAATTTTTGTGTTATGTACGTGTAAGAAAACATAGCTATTATGATATGGAAACTAGGAGTGAAATATGAGGAATTTGTGACTTTTCTGAA
```

A `FASTQ` file normally uses four lines per sequence.

* **Line 1** begins with a '@' character and is followed by a **sequence identifier** and an optional description (like a `FASTA` title line).
* **Line 2** is the **raw sequence letters**.
* **Line 3** begins with a '+' character and is **optionally followed by the same sequence identifier** (and any description) again.
* **Line 4** encodes the **quality values** for the sequence in Line 2, and must contain the same number of symbols as letters in the sequence.

You can find the information of `SRR062634` in the [Sequence Read Archive (SRA) page](https://www.ncbi.nlm.nih.gov/sra/?term=srr062634) of NCBI website.

Here are the quality value characters in left-to-right increasing order of quality (ASCII):

```
!"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~
```

You can learn more about `FASTQ` files on [Wikipedia](https://en.wikipedia.org/wiki/FASTQ_format).

1000 Genome Project at https://www.internationalgenome.org/faq/about-fastq-sequence-read-files/


Many of individuals in 1000 Genome Project have multiple `FASTQ` files, because many of them were sequenced using more than one run of a sequencing machine. \
Each set of files named like `SRR062634_1.filt.fastq.gz`, `SRR062634_2.filt.fastq.gz` and `SRR062634.filt.fastq.gz` represent all the sequence from a sequencing run.

The labels with `_1` and `_2` represent paired-end files, and the files which do not have a number in their name are single-ended reads. (or if one of a pair of reads gets rejected the other read gets placed in the single file.)

### 3. Overall Process

This practice session consists of 4 steps.

1) Preprocess the `FASTQ` file
 * `FastqToSam`
 * `AddOrReplaceReadGroups`
 * `MarkIlluminaAdapters`
 * `SamToFastq`
2) Convert `FASTQ` to `BAM`
 * Map `FASTQ` file to reference with `bwa mem`
 * `MergeBamAlignment`
 * `MarkDuplicates`
 * `SortSam`
 * Base Quality Score Recalibration (`BaseRecalibrator`)
 * `ApplyBQSR`
3) Convert `BAM` to `GVCF`
 * `HaplotypeCaller`
4) Convert `GVCF` to `VCF`
 * `GenotypeGVCFs`

### 4. (Optional) Preprocessing the `FASTQ` file

#### Convert the raw `FASTQ` file to an unmapped `BAM` file

Using `FastqToSam` function of Picard, we can convert the `FASTQ` file to an unmapped `BAM` file.

```
java -jar picard.jar \
F1=SRR062634.filt.fastq \
O=fastq_to_bam_96.bam \
SM=HG00096
```

#### Add read group information in `BAM` file

We can add read groups in `BAM` file by the following command:

```
java -jar picard.jar AddOrReplaceReadGroups \
I=fastq_to_bam_96.bam \
O=add_read_groups_96.bam \
RGID=4 \
RGLB=lib1 \
RGPL=ILLUMINA \
RGPU=unit1 \
RGSM=20
```

#### Mark adapter sequences

Using `MarkIlluminaAdapters`, we can mark adapter sequences.

```
java -Xmx8G -jar picard.jar MarkIlluminaAdapters \
I=add_read_groups_96.bam \
O=mark_adapter_96.bam \
M=mark_adapter_96.metrics.txt
```

#### Convert the preprocessed `BAM` file to a `FASTQ` file

```
java -Xmx8G -jar picard.jar SamToFastq \
I=mark_adapter_96.bam \
FASTQ=fastq_input_96.fq \
CLIPPING_ATTRIBUTE=XT \
CLIPPING_ACTION=2 \
INTERLEAVE=true \
NON_PF=true
```

### 5. Convert the preprocessed `FASTQ` file to an aligned `BAM` file

#### Align reads using `BWA MEM`

GATK's variant discovery workflow recommends Burrows-Wheeler Aligner's maximal exact matches (BWA-MEM) algorithm.

```
bwa mem -M -t 7 -p human_g1k_v37.fasta fastq_input_96.fq > aligned_96.sam
```

You can check the contents of the `SAM` file:

```
@SQ	SN:1	LN:249250621
@SQ	SN:2	LN:243199373
@SQ	SN:3	LN:198022430
...
SRR062634.10000020	16	1	246002445	60	100M	*	0	0	ACAGCACCAGGCCAGCCTTTTTATTTTATTTTAATTTTTATTATTTTGAGACATTCTCGCTCTTTCGCCCAGGCCGGACTGCAGTGGTGCTATCTCAGCT	<C==C:@?A=9:5,=?>=?C=?;EE<CCBCAC=/?@FFAEEEEBEDEEABCFCBBGGEBFEDFCEGFGDF?GAGGGGFGGGGGGGGGGGGFGGGFGGGGG	NM:i:0	MD:Z:100	AS:i:100	XS:i:32
SRR062634.10000713	16	2	39841450	60	100M	*	0	0	TTAGCCATTCTAGTAGCTGTGTAGCAATTATGCTAGTTAACTGGTCAAATCTAATAGAGATGCTATCTAAAATGTGTTATAAAGAATGTGACTTGAGAGT	==:=C@A??DC@CC@CEBCCCCEDEGD=EECEF?EEBEGEFEFEEEGFGEGEGGDGFGFGEDGFGGGGGGGGEGGGGGGGGGGGGGGGGGGGGGGGGGGG	NM:i:0	MD:Z:100	AS:i:100	XS:i:19
SRR062634.10000906	0	5	97444533	60	100M	*	0	0	CAGTTTGATCCTTCTGAATTAGATTTTCCATACATGAAGCCTATGGGACTCTGGTGGGCAGTAGAAGATAAACTGTAATTTAAGTGAGGTTTTTATAAGC	EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE5BDDCC?EEEEEEDBEEEEEEEDEEEEA?DEEEEBE@EEEEADEDEE@AD?E	NM:i:1	MD:Z:48T51	AS:i:95	XS:i:20
```

The header section must be prior to the alignment section if it is present. Headings begin with the '@' symbol, which distinguishes them from the alignment section. \
Alignment sections have 11 mandatory fields, as well as a variable number of optional fields.

The information of some columns are as follows:

* Column 1: Query Template Name (QNAME)
* Column 2: Bitwise Flag 
* Column 3: Reference sequence name (RNAME), often contains the Chromosome name
* Column 4: Leftmost position of where this alignment maps to the reference (POS)
* Column 5: Mapping Quality
* Column 6: Concise Idiosyncratic Gapped Alignment Report (CIGAR) string ([Wikipedia](https://en.wikipedia.org/wiki/Sequence_alignment#Representations))
* Column 10: Segment sequence
* Column 11: ASCII representation of phred-scale base quality

#### Add information to `BAM` file using `MergeBamAlignment`

```
java -Xmx10G -jar picard.jar MergeBamAlignment \
R=human_g1k_v37.fasta \
UNMAPPED=add_read_groups_96.bam \
ALIGNED=aligned_96.sam \
O=preprocessed_96.bam \
CREATE_INDEX=true \
ADD_MATE_CIGAR=true \
CLIP_ADAPTERS=false \
CLIP_OVERLAPPING_READS=true \
INCLUDE_SECONDARY_ALIGNMENTS=true \
MAX_INSERTIONS_OR_DELETIONS=-1  \
PRIMARY_ALIGNMENT_STRATEGY=MostDistant \
ATTRIBUTES_TO_RETAIN=XS
```

#### Mark Duplicates

```
java -jar picard.jar MarkDuplicates \
I=preprocessed_96.bam \
O=mark_dup_96.bam \
M=mark_dup_96.metrics.txt
```

#### Sort, index and convert alignment to a BAM using SortSam

```
java -jar picard.jar SortSam \
I=mark_dup_96.bam \
O=sorted_96.bam \
SO=coordinate 
```

#### Create Recalibration Table using `BaseRecalibrator`

```
gatk --java-options '-Xmx10g' BaseRecalibrator \
-I sorted_96.bam \
-R human_g1k_v37.fasta \
--known-sites ALL.wgs.mergedSV.v8.20130502.svs.genotypes.vcf.gz \
-O recal_data_96.table
```

#### Base Quality Score Recalibration (BQSR)

```
gatk --java-options '-Xmx10g' ApplyBQSR \
-I sorted_96.bam \
-R human_g1k_v37.fasta \
--bqsr-recal-file recal_data_96.table \
-O bqsr_96.bam
```

#### IGV Viewer (Software for visualization of `BAM` file)

We can visualize the aligned `BAM` file with the [IGV viewer](https://software.broadinstitute.org/software/igv/home).

For example, we can observe high coverage around SUMO1P1 gene. (`HG00096.chrom20.ILLUMINA.bwa.GBR.exome.20120522.bam`)

### 6. Converting `BAM` to `GVCF`

#### Convert the individual `BAM` files to `GVCF` files

In this section, we will use the real data of 3 individuals in 1000 Genome Project (`HG00096`, `HG00097`, `HG00099`). \
The aligned `BAM` files of them can be found at `~/GCDA/1_sequencing/data/` folder. \
We can convert these `BAM` files to `GVCF` files.

```
# Convert BAM for the first individual
gatk --java-options "-Xms4g" HaplotypeCaller \
-R ~/GCDA/1_sequencing/data/human_g1k_v37.fasta \
-I ~/GCDA/1_sequencing/data/HG00096.chrom20.ILLUMINA.bwa.GBR.exome.20120522.bam \
-L 20 \
-ERC GVCF \
-O sample01_20.g.vcf
```

```
# Convert BAM for the second individual
gatk --java-options "-Xms4g" HaplotypeCaller \
-R ~/GCDA/1_sequencing/data/human_g1k_v37.fasta \
-I ~/GCDA/1_sequencing/data/HG00097.chrom20.ILLUMINA.bwa.GBR.exome.20130415.bam \
-L 20 \
-ERC GVCF \
-O sample02_20.g.vcf
```

```
# Convert BAM for the third individual
gatk --java-options "-Xms4g" HaplotypeCaller \
-R ~/GCDA/1_sequencing/data/human_g1k_v37.fasta \
-I ~/GCDA/1_sequencing/data/HG00099.chrom20.ILLUMINA.bwa.GBR.exome.20130415.bam \
-L 20 \
-ERC GVCF \
-O sample03_20.g.vcf
```

#### Combine individual `GVCF` files

```
gatk CombineGVCFs \
-R ~/GCDA/1_sequencing/data/human_g1k_v37.fasta \
--variant sample01_20.g.vcf \
--variant sample02_20.g.vcf \
--variant sample03_20.g.vcf \
-O sample_all.g.vcf.gz
```

### 7. Converting `GVCF` to `VCF`

```
gatk --java-options "-Xmx4g" GenotypeGVCFs \
-R human_g1k_v37.fasta \
-V sample_all.g.vcf.gz \
-O sample_all.vcf
```

You can check the contents of the final `VCF` file.

```
##fileformat=VCFv4.2
##ALT=<ID=NON_REF,Description="Represents any possible alternative allele not already represented at this location by REF and ALT">
##FILTER=<ID=LowQual,Description="Low quality">
...
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	HG00096	HG00097	HG00099
20	61795	.	G	T	39.24	.	AC=2;AF=1.00;AN=2;DP=2;ExcessHet=0.0000;FS=0.000;MLEAC=2;MLEAF=1.00;MQ=60.00;QD=19.62;SOR=0.693	GT:AD:DP:GQ:PL	1/1:0,2:2:6:49,6,0	./.:0,0:0:0:0,0,0	./.:0,0:0:0:0,0,0
20	68749	.	T	C	193.81	.	AC=2;AF=0.500;AN=4;DP=11;ExcessHet=0.0000;FS=0.000;MLEAC=2;MLEAF=0.500;MQ=60.00;QD=25.36;SOR=3.611	GT:AD:DP:GQ:PL	1/1:0,5:5:15:208,15,0	./.:0,0:0:0:0,0,0	0/0:4,0:4:12:0,12,115
20	76962	.	T	C	19086.73	.	AC=6;AF=1.00;AN=6;BaseQRankSum=1.80;DP=539;ExcessHet=0.0000;FS=0.000;MLEAC=6;MLEAF=1.00;MQ=59.41;MQRankSum=-3.240e-01;QD=28.73;ReadPosRankSum=1.42;SOR=0.260	GT:AD:DP:GQ:PL	1/1:0,357:357:99:14330,1073,0	1/1:1,65:66:99:2125,188,0	1/1:0,84:84:99:2645,250,0
```

We use this `VCF` file in the analysis!

