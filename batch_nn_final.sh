# set variables
INPUT=$1
OUTFILE1=${INPUT}.txt
OUTFILE2=${INPUT}.png
OUTFILE3=${INPUT}_loss.png
OUTFILE4=${INPUT}_corr.png
OUTFILE5=${INPUT}.root
OUTFILE6=${INPUT}_res.png
OUTFILE7=${INPUT}_gen.png

JOBDIR=${INPUT}
BASEDIR=$HOME/tauregression/CMSSW_8_0_23/src
OUTDIR=$BASEDIR/thesis_plots
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
python $BASEDIR/nn_final.py $INPUT

# copy output back
mkdir -p $OUTDIR
cp $WORKDIR/$OUTFILE1 $OUTDIR/$OUTFILE1
cp $WORKDIR/$OUTFILE2 $OUTDIR/$OUTFILE2
cp $WORKDIR/$OUTFILE3 $OUTDIR/$OUTFILE3
cp $WORKDIR/$OUTFILE4 $OUTDIR/$OUTFILE4
cp $WORKDIR/$OUTFILE5 $OUTDIR/$OUTFILE5
cp $WORKDIR/$OUTFILE6 $OUTDIR/$OUTFILE6
cp $WORKDIR/$OUTFILE7 $OUTDIR/$OUTFILE7

rm -rf $WORKDIR
exit 0
