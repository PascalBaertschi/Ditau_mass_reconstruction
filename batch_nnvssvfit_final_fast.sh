# set variables
INPUT=$1
OUTFILE1=${INPUT}.txt
OUTFILE2=${INPUT}.png
OUTFILE3=${INPUT}_loss.png
OUTFILE4=${INPUT}_delta.png
OUTFILE5=${INPUT}_res.png
OUTFILE6=${INPUT}.root

JOBDIR=${INPUT}
BASEDIR=$HOME/tauregression/CMSSW_8_0_23/src
OUTDIR=$BASEDIR/batch_nnvssvfit_output
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
cd SVfit_standalone/build
export LD_LIBRARY_PATH=${PWD}/../svFit/lib:${LD_LIBRARY_PATH}
cd /shome/pbaertsc/tauregression/CMSSW_8_0_23/src

# run script on scratch
mkdir -p $WORKDIR
cd $WORKDIR
python $BASEDIR/nnvssvfit_final_fast.py $INPUT

# copy output back
mkdir -p $OUTDIR
cp $WORKDIR/$OUTFILE1 $OUTDIR/$OUTFILE1
cp $WORKDIR/$OUTFILE2 $OUTDIR/$OUTFILE2
cp $WORKDIR/$OUTFILE3 $OUTDIR/$OUTFILE3
cp $WORKDIR/$OUTFILE4 $OUTDIR/$OUTFILE4
cp $WORKDIR/$OUTFILE5 $OUTDIR/$OUTFILE5
cp $WORKDIR/$OUTFILE6 $OUTDIR/$OUTFILE6

rm -rf $WORKDIR
exit 0