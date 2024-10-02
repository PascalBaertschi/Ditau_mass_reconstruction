# set variables
INPUT=$1
OUTFILE1=${INPUT}.txt
OUTFILE2=nnoutput_${INPUT}.csv
OUTFILE3=nnoutput_100GeV_${INPUT}.csv
OUTFILE4=nnoutput_110GeV_${INPUT}.csv
OUTFILE5=nnoutput_125GeV_${INPUT}.csv
OUTFILE6=nnoutput_140GeV_${INPUT}.csv
OUTFILE7=nnoutput_180GeV_${INPUT}.csv
OUTFILE8=nnoutput_250GeV_${INPUT}.csv
OUTFILE9=nnoutput_dy_${INPUT}.csv
OUTFILE10=nnoutput_loss_${INPUT}.csv
OUTFILE11=nnoutput_val_loss_${INPUT}.csv


JOBDIR=${INPUT}
BASEDIR=$HOME/tauregression/CMSSW_8_0_23/src
OUTDIR=$BASEDIR/nnoutput
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
python $BASEDIR/nn.py $INPUT

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

rm -rf $WORKDIR
exit 0
