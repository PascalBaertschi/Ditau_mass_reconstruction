# set variables
INPUT=$1
OUTFILE1=${INPUT}.txt
OUTFILE2=${INPUT}.png
OUTFILE3=${INPUT}_loss.png
OUTFILE4=${INPUT}_decaymodes.png
OUTFILE5=${INPUT}_lep.png
OUTFILE6=${INPUT}_semilep.png
OUTFILE7=${INPUT}_had.png
OUTFILE8=${INPUT}_delta.png
OUTFILE9=${INPUT}_res.png
OUTFILE10=${INPUT}_ratio.png
OUTFILE11=${INPUT}_ratiolep.png
OUTFILE12=${INPUT}_ratiosemilep.png
OUTFILE13=${INPUT}_ratiohad.png


JOBDIR=${INPUT}
BASEDIR=$HOME/tauregression/CMSSW_8_0_23/src
OUTDIR=$BASEDIR/batch_output
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
python $BASEDIR/nnvssvfit_split.py $INPUT

# copy output back
mkdir -p $OUTDIR
cp $WORKDIR/$OUTFILE1 $OUTDIR/$OUTFILE1
cp $WORKDIR/$OUTFILE2 $OUTDIR/$OUTFILE2

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
rm -rf $WORKDIR
exit 0