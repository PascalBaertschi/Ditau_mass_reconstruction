# set variables
INPUT=$1
OUTFILE1=${INPUT}.txt
OUTFILE2=${INPUT}.png
OUTFILE3=${INPUT}_loss.png
OUTFILE4=${INPUT}_corr.png
OUTFILE5=${INPUT}.root
OUTFILE6=${INPUT}_res.png
OUTFILE7=${INPUT}_corrres.png
OUTFILE8=${INPUT}_NNinput.png
OUTFILE9=${INPUT}_rescompar.png
OUTFILE10=${INPUT}_profres.png
OUTFILE11=${INPUT}_profabsres.png
OUTFILE12=${INPUT}_rms.png
OUTFILE13=${INPUT}_profrescomp.png
OUTFILE14=${INPUT}_profabsrescomp.png
OUTFILE15=${INPUT}_all.png

JOBDIR=${INPUT}
BASEDIR=$HOME/tauregression/CMSSW_8_0_23/src
OUTDIR=$BASEDIR/final_plots
TOPWORKDIR=/scratch/$USER
WORKDIR=$TOPWORKDIR/$JOBDIR
 

#$ -o /shome/pbaertsc/tauregression/CMSSW_8_0_23/src/batchoutput/
#$ -e /shome/pbaertsc/tauregression/CMSSW_8_0_23/src/batcherrors/

# set CMSSW environment for script
source /mnt/t3nfs01/data01/swshare/psit3/etc/profile.d/cms_ui_env.sh
source $VO_CMS_SW_DIR/cmsset_default.sh
cd /shome/pbaertsc/tauregression/CMSSW_8_0_23/src
eval `scram runtime -sh`
source rootpy_env/bin/activate

# run script on scratch
mkdir -p $WORKDIR
cd $WORKDIR
python $BASEDIR/nn_cuts.py $INPUT

# copy output back
mkdir -p $OUTDIR
cp $WORKDIR/$OUTFILE1 $OUTDIR/$OUTFILE1
cp $WORKDIR/$OUTFILE2 $OUTDIR/$OUTFILE2
cp $WORKDIR/$OUTFILE3 $OUTDIR/$OUTFILE3
cp $WORKDIR/$OUTFILE4 $OUTDIR/$OUTFILE4
cp $WORKDIR/$OUTFILE5 $OUTDIR/$OUTFILE5
cp $WORKDIR/$OUTFILE6 $OUTDIR/$OUTFILE6
cp $WORKDIR/$OUTFILE7 $OUTDIR/$OUTFILE7
cp $WORKDIR/$OUTFILE8 $OUTDIR/$OUTFILE8
cp $WORKDIR/$OUTFILE9 $OUTDIR/$OUTFILE9
cp $WORKDIR/$OUTFILE10 $OUTDIR/$OUTFILE10
cp $WORKDIR/$OUTFILE11 $OUTDIR/$OUTFILE11
cp $WORKDIR/$OUTFILE12 $OUTDIR/$OUTFILE12
cp $WORKDIR/$OUTFILE13 $OUTDIR/$OUTFILE13
cp $WORKDIR/$OUTFILE14 $OUTDIR/$OUTFILE14
cp $WORKDIR/$OUTFILE15 $OUTDIR/$OUTFILE15

rm -rf $WORKDIR
exit 0
