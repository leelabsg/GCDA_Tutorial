# Practice Session #2: Genome-wide Association Studies (GWAS) (April 6, 2023)

In this session, we will learn how to conduct the genome-wide association analysis using SAIGE. \
(References: [Paper](https://www.nature.com/articles/s41588-018-0184-y), [Github](https://github.com/saigegit/SAIGE), [Documentation](https://saigegit.github.io/SAIGE-doc/)) \
This document was created on April 5, 2023 and the following contents were tested on the GSDS cluster (Ubuntu 18.04 LTS).

### 1. Setting up the environment

We will use [Docker](https://www.docker.com/) on the GSDS cluster (`leelabsg00` node). \
It is already created on the GSDS cluster, but you can create the environment on your local machine with the following command:

```
# Pull SAIGE docker image
sudo docker pull wzhou88/saige:1.1.6.3

# Give previlege to use docker without sudo
sudo usermod -aG docker $USER
```

You can test if it works:

```
docker run --rm -it -v /home/n1/leelabguest/GCDA/2_GWAS:/home/n1/leelabguest/GCDA/2_GWAS wzhou88/saige:1.1.6.3 /bin/bash
```

### 2. Preparing data

We need (individual-level) genotype and phenotype files to conduct GWAS. \
However, access to these files is strictly restricted.

Let's take a quick look at what the real data (UK Biobank) looks like.

#### Phenotype file

The phenotype file of UK Biobank looks like the following:

```
f.eid f.46.0.0 f.47.0.0
1000001 18 21
1000002 32 44
1000003 45 42
1000004 42 44
1000005 51 48
1000006 53 45
1000007 28 30
1000008 29 27
1000009 33 32
```

You can find the information of each phenotype column in [UK Biobank showcase](https://biobank.ndph.ox.ac.uk/showcase/).

* Field ID 46 means 'hand grip strength (left)'
* Field ID 47 means 'hand grip strength (right)'

#### Genotype file

##### Variant Call Format (`vcf`)

It contains meta-information lines, a header line, and then data lines each containing information about a position in the genome.


##### PLINK binary (`bed`, `bim`, `fam`)

PLINK binary files are the binary version of PLINK files (`ped`, `map`). \
PLINK binary genotype data consists of 3 files:

* `bed` file: Genotype data in binary format
* `bim` file: Variants information
* `fam` file: Sample information

`bim` file contains the information of variants.

```
1	rs1	0	1	C	A
1	rs2	0	2	C	A
1	rs3	0	3	C	A
1	rs4	0	4	C	A
1	rs5	0	5	C	A
1	rs6	0	6	C	A
1	rs7	0	7	C	A
1	rs8	0	8	C	A
1	rs9	0	9	C	A
1	rs10	0	10	C	A
```

`fam` file contains the information of samples.

```
1a1	1a1	0	0	0	-9
1a2	1a2	0	0	0	-9
1a3	1a3	0	0	0	-9
1a4	1a4	0	0	0	-9
1a5	1a5	0	0	0	-9
1a6	1a6	0	0	0	-9
1a7	1a7	0	0	0	-9
1a8	1a8	0	0	0	-9
1a9	1a9	0	0	0	-9
1a10	1a10	0	0	0	-9
```

`bed` file contains the genotype information (in binary format). \
We cannot easily check the genotype information, but we can see the bitwise 

```
00000000: 01101100 00011011 00000001 11111111 11111111 11111111  l.....
00000006: 11111111 11111111 11111111 11111111 11111111 11111111  ......
0000000c: 11111111 11111111 11111111 11111111 11111111 11111111  ......
00000012: 11110111 11110111 11111111 11111111 11111111 11111110  ......
00000018: 11111111 11111111 11111111 11111111 11111111 11111111  ......
0000001e: 11111111 11111111 11111111 11111111 11111111 11111111  ......
00000024: 11111111 11111111 11111111 11111111 11111111 11111111  ......
0000002a: 11111111 11111111 11111111 11111111 11111111 11111111  ......
00000030: 11111111 11111111 11111111 11111111 11110111 11111111  ......
00000036: 11111111 11111111 11111111 11111111 11111111 11111111  ......
0000003c: 11111111 11111111 11111111 11111111 11111111 11111111  ......
00000042: 11111111 11111111 11111111 11111111 11111111 11111111  ......
00000048: 11111111 11110011 11111111 11111111 11111111 11111111  ......
0000004e: 11111111 11111111 11111111 11111111 11111111 11111111  ......
00000054: 11111111 11111111 11111111 11111111 11111111 11111111  ......
```

You can find more details on [PLINK website](https://zzz.bwh.harvard.edu/plink/binary.shtml).


#### Example Data

We will use `pheno_1000samples.txt_withdosages_withBothTraitTypes.txt` as a phenotype file.

```
y_quantitative y_binary x1 x2 IID a1 a2 a3 a4 a5 a6 a7 a8 a9 a10
2.0046544617651 0 1.51178116845085 1 1a1 0 0 0 0 0 0 0 0 1 0
0.104213400269085 0 0.389843236411431 1 1a2 0 0 0 0 0 0 0 0 1 1
-0.397498354133647 0 -0.621240580541804 1 1a3 0 0 0 0 0 0 0 0 0 1
-0.333177899030597 0 -2.2146998871775 1 1a4 0 0 0 0 0 0 0 0 1 1
1.21333962248852 0 1.12493091814311 1 1a5 0 0 0 0 0 0 0 0 1 0
-0.275411643032321 0 -0.0449336090152309 1 1a6 0 0 0 0 0 0 0 0 1 0
0.438532936074923 0 -0.0161902630989461 0 1a7 0 0 0 0 0 0 0 0 0 0
0.0162938047248591 0 0.943836210685299 0 1a8 0 0 0 0 0 0 0 0 1 1
0.147167262428064 0 0.821221195098089 1 1a9 0 0 0 0 0 0 0 0 1 0
```





### 3. Run GWAS using SAIGE

#### What is SAIGE?

* Mixed effect model-based method
* Score test to compute test statistics
* Several techniques for fast computation

#### Why SAIGE?

* Can account for related individuals
* Fast computation
* Prevent type I error inflation in case of unbalanced case-control ratio

#### Process

##### Step 0. (Optional) Create sparse GRM

This sparse GRM only needs to be created once for each data set, e.g. a biobank, and can be used for all different phenotypes as long as all tested samples are in the sparse GRM.

```
docker run -v /home/n1/leelabguest/GCDA:/home/n1/leelabguest/GCDA wzhou88/saige:1.1.6.3 createSparseGRM.R \
        --plinkFile=/home/n1/leelabguest/GCDA/2_GWAS/SAIGE/extdata/input/nfam_100_nindep_0_step1_includeMoreRareVariants_poly \
        --nThreads=4 \
        --outputPrefix=/home/n1/leelabguest/GCDA/usr/YOUR_DIRECTORY/sparseGRM \
        --numRandomMarkerforSparseKin=2000 \
        --relatednessCutoff=0.125
```

##### Step 1. Fitting the null (logistic/linear) mixed model

###### Binary trait example

```
docker run -v /home/n1/leelabguest/GCDA:/home/n1/leelabguest/GCDA wzhou88/saige:1.1.6.3 step1_fitNULLGLMM.R \
        --plinkFile=/home/n1/leelabguest/GCDA/2_GWAS/SAIGE/extdata/input/nfam_100_nindep_0_step1_includeMoreRareVariants_poly_22chr \
        --phenoFile=/home/n1/leelabguest/GCDA/2_GWAS/SAIGE/extdata/input/pheno_1000samples.txt_withdosages_withBothTraitTypes.txt \
        --phenoCol=y_binary \
        --covarColList=x1,x2,a9,a10 \
        --qCovarColList=a9,a10 \
        --sampleIDColinphenoFile=IID \
        --traitType=binary \
        --outputPrefix=/home/n1/leelabguest/GCDA/usr/YOUR_DIRECTORY/example_binary \
        --nThreads=8 \
        --IsOverwriteVarianceRatioFile=TRUE
```

###### Quantitative trait example

```
docker run -v /home/n1/leelabguest/GCDA:/home/n1/leelabguest/GCDA wzhou88/saige:1.1.6.3 step1_fitNULLGLMM.R \
        --plinkFile=/home/n1/leelabguest/GCDA/2_GWAS/SAIGE/extdata/input/nfam_100_nindep_0_step1_includeMoreRareVariants_poly_22chr \
        --useSparseGRMtoFitNULL=FALSE \
        --phenoFile=/home/n1/leelabguest/GCDA/2_GWAS/SAIGE/extdata/input/pheno_1000samples.txt_withdosages_withBothTraitTypes.txt \
        --phenoCol=y_quantitative \
        --covarColList=x1,x2,a9,a10 \
        --qCovarColList=a9,a10 \
        --sampleIDColinphenoFile=IID \
        --invNormalize=TRUE \
        --traitType=quantitative \
        --outputPrefix=/home/n1/leelabguest/GCDA/usr/YOUR_DIRECTORY/example_quantitative \
        --nThreads=8 \
        --IsOverwriteVarianceRatioFile=TRUE
```

You can find more example usages in [SAIGE Documentation page](https://saigegit.github.io/SAIGE-doc/docs/single_example.html).

##### Step 2. Performing single-variant association tests

###### Binary trait VCF example

```
docker run -v /home/n1/leelabguest/GCDA:/home/n1/leelabguest/GCDA wzhou88/saige:1.1.6.3 step2_SPAtests.R \
        --vcfFile=/home/n1/leelabguest/GCDA/2_GWAS/SAIGE/extdata/input/genotype_100markers.vcf.gz \
        --vcfFileIndex=/home/n1/leelabguest/GCDA/2_GWAS/SAIGE/extdata/input/genotype_100markers.vcf.gz.csi \
        --vcfField=GT \
        --SAIGEOutputFile=/home/n1/leelabguest/GCDA/usr/YOUR_DIRECTORY/genotype_100markers_marker_vcf.txt \
        --chrom=1 \
        --minMAF=0 \
        --minMAC=20 \
        --GMMATmodelFile=/home/n1/leelabguest/GCDA/usr/YOUR_DIRECTORY/example_binary.rda \
        --varianceRatioFile=/home/n1/leelabguest/GCDA/usr/YOUR_DIRECTORY/example_binary.varianceRatio.txt
```

###### Binary trait PLINK example

```
docker run -v /home/n1/leelabguest/GCDA:/home/n1/leelabguest/GCDA wzhou88/saige:1.1.6.3 step2_SPAtests.R \
        --bedFile=/home/n1/leelabguest/GCDA/2_GWAS/SAIGE/extdata/input/genotype_100markers.bed \
        --bimFile=/home/n1/leelabguest/GCDA/2_GWAS/SAIGE/extdata/input/genotype_100markers.bim \
        --famFile=/home/n1/leelabguest/GCDA/2_GWAS/SAIGE/extdata/input/genotype_100markers.fam \
        --AlleleOrder=alt-first \
        --SAIGEOutputFile=/home/n1/leelabguest/GCDA/usr/YOUR_DIRECTORY/genotype_100markers_marker_plink.txt \
        --chrom=1 \
        --minMAF=0 \
        --minMAC=20 \
        --GMMATmodelFile=/home/n1/leelabguest/GCDA/usr/YOUR_DIRECTORY/example_binary.rda \
        --varianceRatioFile=/home/n1/leelabguest/GCDA/usr/YOUR_DIRECTORY/example_binary.varianceRatio.txt
```


###### Quantitative trait example

```
docker run -v /home/n1/leelabguest/GCDA:/home/n1/leelabguest/GCDA wzhou88/saige:1.1.6.3 step2_SPAtests.R \
        --vcfFile=/home/n1/leelabguest/GCDA/2_GWAS/SAIGE/extdata/input/genotype_100markers.vcf.gz \
        --vcfFileIndex=/home/n1/leelabguest/GCDA/2_GWAS/SAIGE/extdata/input/genotype_100markers.vcf.gz.csi \
        --vcfField=GT \
        --SAIGEOutputFile=/home/n1/leelabguest/GCDA/usr/YOUR_DIRECTORY/genotype_100markers_marker_quant.txt \
        --chrom=1 \
        --minMAF=0 \
        --minMAC=20 \
        --GMMATmodelFile=/home/n1/leelabguest/GCDA/usr/YOUR_DIRECTORY/example_quantitative.rda \
        --varianceRatioFile=/home/n1/leelabguest/GCDA/usr/YOUR_DIRECTORY/example_quantitative.varianceRatio.txt
```


### 4. Drawing plots

Using `qqman` package in R, we can draw manhattan plot and Q-Q plot with the GWAS result (summary statistics).

```
# Draw manhattan and Q-Q plot (r_env)
library(qqman)
library(data.table)
gwas <- fread("/home/n1/leelabguest/GCDA/2_GWAS/sample.txt.gz", header=T)

man_file <- '/home/n1/leelabguest/GCDA/usr/YOUR_DIRECTORY/manhattan_plot.png'
qq_file <- '/home/n1/leelabguest/GCDA/usr/YOUR_DIRECTORY/qq_plot.png'

colnames(gwas)[c(1, 2, 3, 13)] <- c('CHR', 'BP', 'SNP', 'P')
png(file=man_file, width=1000, height=500)
manhattan(gwas, main='Manhattan Plot')
dev.off()

png(file=qq_file, width=1000, height=1000)
qq(gwas$P, main='Q-Q plot')
dev.off()
```

You can copy files to your local machine (from the server) using `scp` command.

```
scp 'leelabguest@147.47.200.192:SOURCE_PATH' DESTINATION_PATH

# Example
scp 'leelabguest@147.47.200.192:~/GCDA/usr/YOUR_DIRECTORY/*.png' .
```

Or you can use [LocusZoom](http://locuszoom.org/) or [PheWeb](https://github.com/statgen/pheweb) to visualize your GWAS results and host them on the web server.
