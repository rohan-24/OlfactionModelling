/*

Project description

*/


#include <stdlib.h>
#include <vector>
#include <string>
#include <numeric>
#include <random>
#include <limits>
#include <cstring>
#include <map>
#include <fstream>
#include <sstream>
#include <set>
#include <algorithm>
#include <filesystem>
#include "bcpnnsim.h"

using namespace std;
using namespace Globals;


/*

TODO: GET NETWORK TO RUN WITHOUT HYPERCOLUMNS

Fix Actplot display and try to collect statistics of recall with reducing number of active HCs
*/



long seed = 0;
int H = 15,M = 15,H2=15,M2=15;
float igain = 1,again = 2,taum = 0.001,nmean = 0,namp = 0,taua = 0.200,adgain = 0,taup = 1,taucond = timestep, bgain = 1,
    bwgain = 1,pdens = 1,wdens = 1,cspeed = 1,recuwgain = 0,assowgain = 1, p0 = 1, tausd = 0, tausf = 0,assocpdens=1, 
    epsfloat_multiplier = 1,assoc_cspeed=1,assoc_dist=0,thres=0;
int nstep = 1,ngap = 1,npat1 = 1,npat2 = 1,etrnrep = 1,atrnrep = 1, recallnstep=100,recallngap=100;
int biased_item = -1, biased_context = -1, biased_assoc = -1,cueHCs = 5,distortHCs1=0,distortHCs2=0,use_intensity = 0,use_trneffort=0,use_familiarity=0;
string etrn_bias_mode = "none"; // for training LTM1 and 2. stim_length, kappa, none
string pattstr = "ortho",modestr = "resetstate",wgainstr = "onlyassoc", runflag = "full", encode_order="normal",cued_net="LTM1",partial_mode="uniform";
Pat_t pattype = ORTHO;
vector<vector<int> > patstat,ltm1_items_counts,ltm2_items_counts;
vector<int> simstages,patstat_ltm1_col,patstat_ltm2_col;
map <int,int> odintensity_HCs;
vector<vector<float>> trpats1,trpats2,trnphase_partialpats,recallphase_partialpats;

PopH *ltm1 = NULL;
PrjH *prj11 = NULL;

PopH *ltm2 = NULL;
PrjH *prj22 = NULL,*prj12 = NULL,*prj21 = NULL;

int subjects = 0, odors = 0, patsize=0, patsize1 = 0, patsize2 = 0;

void parseparams(std::string paramfile) {

    Parseparam *parseparam = new Parseparam(paramfile);

    parseparam->postparam("seed",&seed,Long);
    parseparam->postparam("H",&H,Int);
    parseparam->postparam("M",&M,Int);
    parseparam->postparam("H2",&H2,Int);
    parseparam->postparam("M2",&M2,Int);
    parseparam->postparam("timestep",&timestep,Float);
    parseparam->postparam("igain",&igain,Float);
    parseparam->postparam("again",&again,Float);
    parseparam->postparam("taum",&taum,Float);
    parseparam->postparam("taua",&taua,Float);
    parseparam->postparam("adgain",&adgain,Float);
    parseparam->postparam("nmean",&nmean,Float);
    parseparam->postparam("namp",&namp,Float);
    parseparam->postparam("pdens",&pdens,Float);
    parseparam->postparam("assocpdens",&assocpdens,Float);
    parseparam->postparam("wdens",&wdens,Float);
    parseparam->postparam("p0",&p0,Float);
    parseparam->postparam("tausd",&tausd,Float);
    parseparam->postparam("tausf",&tausf,Float);
    parseparam->postparam("cspeed",&cspeed,Float);
    parseparam->postparam("assoc_cspeed",&assoc_cspeed,Float);
    parseparam->postparam("assoc_dist",&assoc_dist,Float);
    parseparam->postparam("taucond",&taucond,Float);
    parseparam->postparam("taup",&taup,Float);
    parseparam->postparam("recuwgain",&recuwgain,Float);
    parseparam->postparam("assowgain",&assowgain,Float);
    parseparam->postparam("bgain",&bgain,Float);
    parseparam->postparam("bwgain",&bwgain,Float);
    parseparam->postparam("pattstr",&pattstr,String);
    parseparam->postparam("etrnrep",&etrnrep,Int);
    parseparam->postparam("atrnrep",&atrnrep,Int);
    parseparam->postparam("nstep",&nstep,Int);
    parseparam->postparam("recallnstep",&recallnstep,Int);
    parseparam->postparam("recallngap",&recallngap,Int);
    parseparam->postparam("ngap",&ngap,Int);
    parseparam->postparam("thres",&thres,Float);
    parseparam->postparam("encode_order",&encode_order,String);
    parseparam->postparam("runflag",&runflag,String);
    parseparam->postparam("epsfloat_multiplier",&epsfloat_multiplier,Float);
    parseparam->postparam("cued_net",&cued_net,String);
    parseparam->postparam("partial_mode",&partial_mode,String);
    parseparam->postparam("cueHCs",&cueHCs,Int);
    parseparam->postparam("distortHCs1",&distortHCs1,Int);
    parseparam->postparam("distortHCs2",&distortHCs2,Int);
    parseparam->postparam("use_intensity",&use_intensity,Int);
    parseparam->postparam("use_familiarity",&use_familiarity,Int);
    parseparam->postparam("use_trneffort",&use_trneffort,Int);
    parseparam->doparse();

}


void setpattype(string pattstr) {

    if (pattstr=="ortho" || pattstr=="rndseq")
	pattype = ORTHO;
    else if (pattstr=="rnd01")
	pattype = RND01;
    else
	error("olflangmain1::setpatt","Illegal pattstr");

}


void setwgain(string wgainstr) {


    if (wgainstr=="alla") {

    prj11->setparam(WGAIN,recuwgain); prj22->setparam(WGAIN,recuwgain);
    prj12->setparam(WGAIN,assowgain); prj21->setparam(WGAIN,assowgain);
    prj11->setparam(BGAIN,bgain*1.2); prj22->setparam(BGAIN,bgain*1.);
    prj12->setparam(BGAIN,bgain*1.); prj21->setparam(BGAIN,bgain*1.2);

    } else if (wgainstr=="zero") {

    prj11->setparam(WGAIN,0); prj22->setparam(WGAIN,0);
    prj12->setparam(WGAIN,0); prj21->setparam(WGAIN,0);
	

    } else if (wgainstr=="recu") {

    // prj11->setparam(EWGAIN,recuwgain); prj11->setparam(IWGAIN,recuwgain/4); 
    // prj22->setparam(EWGAIN,recuwgain); prj22->setparam(IWGAIN,recuwgain*2); 
    prj11->setparam(WGAIN,recuwgain); prj22->setparam(WGAIN,recuwgain);
    prj12->setparam(WGAIN,0); prj21->setparam(WGAIN,0);

  }   else if (wgainstr=="asso") {

    prj11->setparam(WGAIN,0); prj22->setparam(WGAIN,0);
    prj12->setparam(WGAIN,assowgain); prj21->setparam(WGAIN,assowgain);

    } else if (wgainstr=="asso12") {

    prj11->setparam(WGAIN,0); prj22->setparam(WGAIN,0);
    prj12->setparam(WGAIN,assowgain); prj21->setparam(WGAIN,0);

    } else if (wgainstr=="asso21") {

    prj11->setparam(WGAIN,recuwgain); prj22->setparam(WGAIN,recuwgain);
    prj12->setparam(WGAIN,0); prj21->setparam(WGAIN,assowgain);
    // prj21->setparam(EWGAIN,recuwgain); prj21->setparam(IWGAIN,recuwgain/4); 
    // prj12->setparam(EWGAIN,recuwgain); prj12->setparam(IWGAIN,recuwgain/4); 

    }  else error("emmain1::setwgain","Illegal wgainstr");


}


void resetstate() {

    Pop::resetstateall(); Prj::resetstateall();
     // ltm1->resetstate(); prj11->resetstate();

}


void setmode(string modestr,string wgainstr = "zero") {

    if (modestr=="encode") {

	ltm1->setparam(BWGAIN,0); ltm2->setparam(BWGAIN,0);

    ltm1->setparam(AGAIN,again); ltm2->setparam(AGAIN,again);

    ltm1->setparam(ADGAIN,0); ltm2->setparam(ADGAIN,0);

	prj11->setparam(PRN,1); prj22->setparam(PRN,1); prj12->setparam(PRN,1); prj21->setparam(PRN,1);

    } else if (modestr=="encoderecu") {

	ltm1->setparam(BWGAIN,0); ltm2->setparam(BWGAIN,0);

    ltm1->setparam(ADGAIN,0); ltm2->setparam(ADGAIN,0);

	ltm1->setparam(AGAIN,again); ltm2->setparam(AGAIN,again);

	prj11->setparam(PRN,1); prj22->setparam(PRN,1); prj12->setparam(PRN,0); prj21->setparam(PRN,0);

    } else if (modestr=="encodeasso") {

	ltm1->setparam(BWGAIN,0); ltm2->setparam(BWGAIN,0);

	ltm1->setparam(AGAIN,again); ltm2->setparam(AGAIN,again);

    ltm1->setparam(ADGAIN,0); ltm2->setparam(ADGAIN,0);

	prj11->setparam(PRN,0); prj22->setparam(PRN,0); prj12->setparam(PRN,1); prj21->setparam(PRN,1);

    } else if (modestr=="recall") {

	ltm1->setparam(BWGAIN,bwgain); ltm2->setparam(BWGAIN,bwgain);

	ltm1->setparam(ADGAIN,adgain); ltm2->setparam(ADGAIN,adgain);

	ltm1->setparam(AGAIN,again); ltm2->setparam(AGAIN,again);

    ltm1->setparam("thres",thres);   ltm2->setparam("thres",thres);

	prj11->setparam(PRN,0); prj22->setparam(PRN,0); prj12->setparam(PRN,0); prj21->setparam(PRN,0);

    } else if (modestr=="resetstate") resetstate();

    else error("olflangmain1::setmode","Illegal modestr");

    setwgain(wgainstr);

}

// Function to calculate the mean of a 1D vector of floats
float calculateMean(const vector<float>& vec) {
    float sum = accumulate(vec.begin(), vec.end(), 0.0f);
    float mean = sum / vec.size();
    return mean;
}

// Function to calculate the mean of a 2D vector of floats
float calculateMean(const vector<vector<float>>& vec) {
    float sum = 0.0f;
    size_t count = 0;

    for (const auto& row : vec) {
        sum += accumulate(row.begin(), row.end(), 0.0f);
        count += row.size();
    }

    float mean = sum / count;
    return mean;
}

/***** Use for custom initiation P-traces and Wij/Bj */


void initPWB(string prj, vector<vector<float> > w,int HC,int MC) {

  float P = 1;
  vector<float> Pi(HC*MC),Pj(HC*MC);
  vector<vector<float> > Pij(HC*MC,vector<float>(HC*MC));

  for (int i=0; i<HC*MC; i++) Pi[i] = gnextfloat();

  for (int j=0; j<HC*MC; j++) Pj[j] = gnextfloat();

  for (int i=0; i<HC*MC; i++)

    for (int j=0; j<HC*MC; j++) 
        Pij[i][j] = Pi[i]*Pj[j]/P * exp(w[i][j]);


    if(prj == "prj11")
    {
    prj11->setstate("P",P);
    prj11->setstate("Pi",Pi);
    prj11->setstate("Pj",Pj);
    prj11->setstate("Pij",Pij);
    }

    else if(prj == "prj12")
    {
    prj12->setstate("P",P);
    prj12->setstate("Pi",Pi);
    prj12->setstate("Pj",Pj);
    prj12->setstate("Pij",Pij);
    }

    else if(prj == "prj21")
    {
    prj21->setstate("P",P);
    prj21->setstate("Pi",Pi);
    prj21->setstate("Pj",Pj);
    prj21->setstate("Pij",Pij);
    }

    else if(prj == "prj22")
    {
    prj22->setstate("P",P);
    prj22->setstate("Pi",Pi);
    prj22->setstate("Pj",Pj);
    prj22->setstate("Pij",Pij);
    }

    else error("olflangmain:initPWB","illegal projection");


}

void initPWB(string prj, vector<vector<float> > w, vector<float> b ,int Ni,int Nj) {

/*
Note: Be careful about following right i and j notations when passing n1 and n2
*/
  float P = 1;
  vector<float> Pi(Ni),Pj(Nj);
  vector<vector<float> > Pij(Ni,vector<float>(Nj));

  // std::cout<<"\n"<<prj;
  // std::cout<<"\n Pi Size: ["<<Pi.size()<<"]";
  // std::cout<<"\n Pij Size: ["<<Pj.size()<<"]";
  // std::cout<<"\n Pij Size: ["<<Pij.size()<<","<<Pij[0].size()<<"]";

  for (int i=0; i<Ni; i++) Pi[i] = gnextfloat(); 

  for (int j=0; j<Nj; j++) Pj[j] = exp(b[j]);//gnextfloat();

  for (int i=0; i<Ni; i++)

    for (int j=0; j<Nj; j++) {

        Pij[i][j] = Pi[i]*Pj[j]/P * exp(w[i][j]);

        // printf("P = %.4f Pi[i] = %.4f Pj[j] = %.4f Pij[i,j] = %.4f Wij[i,j] = %.4f\n",
        //     P,Pi[i],Pj[j],Pij[i][j],log(P*Pij[i][j]/Pi[i]/Pj[j]));

    }

    if(prj == "prj11")
    {
    prj11->setstate("P",P);
    prj11->setstate("Pi",Pi);
    prj11->setstate("Pj",Pj);
    prj11->setstate("Pij",Pij);
    }

    else if(prj == "prj12")
    {
    prj12->setstate("P",P);
    prj12->setstate("Pi",Pi);
    prj12->setstate("Pj",Pj);
    prj12->setstate("Pij",Pij);
    }

    else if(prj == "prj21")
    {
    prj21->setstate("P",P);
    prj21->setstate("Pi",Pi);
    prj21->setstate("Pj",Pj);
    prj21->setstate("Pij",Pij);
    }

    else if(prj == "prj22")
    {
    prj22->setstate("P",P);
    prj22->setstate("Pi",Pi);
    prj22->setstate("Pj",Pj);
    prj22->setstate("Pij",Pij);
    }

    else error("emmain:initPWB","illegal projection");


}
/*********************************************************/


void setuplogging() {

    ltm1->logstate("inp","inp1.log");
    ltm1->logstate("dsup","dsup1.log");
    ltm1->logstate("act","act1.log");
    ltm1->logstate("bwsup","bwsup1.log");
    ltm1->logstate("ada","ada1.log");
    ltm1->logstate("expdsup","expdsup1.log");

    ltm2->logstate("inp","inp2.log");
    ltm2->logstate("dsup","dsup2.log");
    ltm2->logstate("act","act2.log");
    ltm2->logstate("bwsup","bwsup2.log");
    ltm2->logstate("ada","ada2.log");
    ltm2->logstate("expdsup","expdsup2.log");

    ltm1->logstate("H_en","H_en1.log");
    ltm2->logstate("H_en","H_en2.log");

    // if (prj11!=NULL) prj11->logstate("Bj","Bj11.log");
    // if (prj11!=NULL) prj11->logstate("Wij","Wij11.log");

    // if (prj22!=NULL) prj22->logstate("Bj","Bj22.log");
    // if (prj22!=NULL) prj22->logstate("Wij","Wij22.log");

    // if (prj12!=NULL) prj12->logstate("Bj","Bj12.log");
    // if (prj12!=NULL) prj12->logstate("Wij","Wij12.log");

    // if (prj21!=NULL) prj21->logstate("Bj","Bj21.log");
    // if (prj21!=NULL) prj21->logstate("Wij","Wij21.log");

    // if (prj11!=NULL) prj11->logstate("Pij","Pij11.log");
    // if (prj11!=NULL) prj11->logstate("Pi","Pi11.log");
    // if (prj11!=NULL) prj11->logstate("Pj","Pj11.log");   

    // if (prj22!=NULL) prj22->logstate("Pij","Pij22.log");
    // if (prj22!=NULL) prj22->logstate("Pi","Pi22.log");
    // if (prj22!=NULL) prj22->logstate("Pj","Pj22.log"); 
    // if (prj11!=NULL) prj11->logstate("P","P11.log");


}


void getpatstat(string filename) {

    patstat = readtxtipats(filename) ;

    if (patstat.size()==0) return;
    if (encode_order == "random")
    {
      //auto engine = std::default_random_engine{};
      std::random_device rd;
      std::mt19937 g(rd());
      std::shuffle(std::begin(patstat),std::end(patstat),g);
    }

    int imax1 = 0,imax2 = 0;

    for (size_t i=0; i<patstat.size(); i++) {

	if (patstat[i][0]>imax1) imax1 = patstat[i][0];
	if (patstat[i][1]>imax2) imax2 = patstat[i][1];

    patstat_ltm1_col.push_back(patstat[i][0]);
    patstat_ltm2_col.push_back(patstat[i][1]);
    }

    npat1 = imax1 + 1;

    npat2 = imax2 + 1;

    // Get number of repetitions of each item in both columns of patstat
    int count;
    vector <int> temp;

    for (int i = 0;i<npat1;i++) {
        count = std::count(patstat_ltm1_col.begin(), patstat_ltm1_col.end(), i);
        temp = {i,count};
        ltm1_items_counts.push_back(temp);
    }

    for (int i = 0;i<npat2;i++) {
        count = std::count(patstat_ltm2_col.begin(), patstat_ltm2_col.end(), i);
        temp = {i,count};
        ltm2_items_counts.push_back(temp);
    }

}

void init_intensityHCsmap(string filename) {
    /*
        Initialize map containing number of HCs to deactivate in odor patterns (based on odor intensity)
    */


    std::vector <std::vector <int>> odintensity_vec;
    odintensity_vec = readtxtipats(filename);

    for (size_t i=0; i<odintensity_vec.size(); i++) {
        odintensity_HCs[odintensity_vec[i][0]] = odintensity_vec[i][1];
    }

    for (auto &pair: odintensity_HCs) {
        std::cout << "{" << pair.first << ": " << pair.second << "}\n";
    }

}

int get_odintensityHCs(int od) {

    auto it = odintensity_HCs.find(od); 
    
    if (it != odintensity_HCs.end())
        return(it->second);

    else
        throw std::runtime_error("get_odintensityHCs(): odor not found");
}

void fwritevec (vector<int> v, string filename)
{
    std::ofstream outf;
    outf.open(filename.c_str(),ios::binary);
    for (auto val : v) {
        outf<<val<<std::endl;
        if (outf.bad()) {
            throw std::runtime_error("Failed to write to outfile!");
        }
    }
    outf.close();
}



// vector<vector<float> > readbinmat(string filename, int n1,int n2)
// {
//   //read matrix from .bin file
//   // int m = n*n;
//   vector<float> state; //1d vector
//   state.resize(n2);
//   vector<vector<float> > state2d; //2d vector of size NxN
//   state2d.resize(n1);
//   for(int i = 0;i<n1;i++)
//   {
//     state2d[i].resize(n2);
//   }
//   ifstream inf;
//   inf.open(filename.c_str(),ios::binary);
//   if(!inf)
//     error("emmain::readbinmat","Could not open file: " + filename);
//   inf.read(reinterpret_cast<char *>(&state[0]), n2*sizeof(state[0]));
//   float f;
//   for(int i = 0; i < state.size(); i++)
//   {
//       int row = i/n1;
//       int col = i%n2;
//       state2d[row][col] = state[i];
//   }
//   inf.close();
//   return state2d;
// }

std::vector<std::vector<float>> readbinmat(string filename, int N1, int N2)
{
    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open())
    {
        std::cout << "Failed to open file: " << filename << std::endl;
        return {};
    }

    std::vector<float> data(N1 * N2);
    file.read((char *)data.data(), N1 * N2 * sizeof(float));

    std::vector<std::vector<float>> result(N1);
    for (int i = 0; i < N1; i++)
    {
        result[i].resize(N2);
        memcpy(result[i].data(), &data[i * N2], N2 * sizeof(float));
    }

    return result;
}

vector<float> readbinvec(string filename, int n)
{
  //Read vector from .bin file
  vector<float> state (n);
  ifstream inf;
  inf.open(filename.c_str(),ios::binary);
  if(!inf)
    error("Globals::readbinvec","Could not open file: " + filename);
  inf.read(reinterpret_cast<char *>(&state[0]), n*sizeof(state[0]));
  inf.close();

  return state;
}

void setPRN(int val) {

    prj11->setparam(PRN,val); prj22->setparam(PRN,val); prj12->setparam(PRN,val); prj21->setparam(PRN,val);

}




vector<vector<vector<float>>>  getpats(string fname, int single_sub_flag) {

  vector<float> state; //1d vector
  // state.resize(subjects);
  
  // float state[subjects];
  ifstream inf;
  inf.open(fname.c_str(),ios::in);

  if(!inf)
    error("olflangmain::getpats","Could not open file: " + fname);

  // inf.read(reinterpret_cast<char *>(&state[0]), subjects*sizeof(uint32_t));
  float element;
  int counter = 0;
  if (single_sub_flag = 1)
  {
    while (inf >> element)
    {   
        if (counter == 0) {

            odors = int(element);
        }
        else if (counter == 1)
            patsize = int(element);
        else
            state.push_back(element);

        counter++;
    }
    subjects = 1;
  }

  else
  {
    while (inf >> element)
    {   
        if (counter == 0)
            subjects = int(element);
        else if (counter == 1)
            odors = int(element);
        else if (counter == 2)
            patsize = int(element);
        else
            state.push_back(element);

        counter++;
    }

  }

  inf.close();


  vector< vector< vector<float>>> state3d(subjects , vector< vector<float> > (odors, vector<float> (patsize) ) ); //3d vector subjects x odors x patsize

  for(int i = 0; i < subjects; i++)
  {
  	for(int j = 0; j < odors; j++)
  	{
  		for(int k = 0; k < patsize; k++)
  	
      		state3d[i][j][k] = state[(i*odors*patsize)+(j*patsize)+k];
      }
  }

  return state3d;

}

vector<vector<float>> readbinpats(string fname,int npats,int patlen,string net_type) 
{
    /**
     Read bin file to extract patterns
    **/
    vector<float> state;
    vector<vector<float>> p(npats, vector<float> (patlen));
    float element;
    // ifstream inf;
    // inf.open(fname.c_str(),ios::in);

    std::ifstream inf(fname.c_str(), std::ios::binary);

    if(!inf)
        error("olflangmain::getpats","Could not open file: " + fname);

    while (inf.read(reinterpret_cast<char*>(&element), sizeof(float)))
    {   
        state.push_back(element);
        
    }


    for(int i = 0; i < npats; i++)
    {   
        for(int j = 0; j < patlen; j++) {
            p[i][j] = state[(i*patlen)+j];
      }
    }

    if (net_type=="od")
        patsize2=patlen;

    return p;

}

float getnoiseperturb(float stddev=0.3) {
    /*
        Return a random float value to act as perturbation
    */

    ///// Uniform distributed noise
    // std::random_device rd;  // Will be used to obtain a seed for the random number engine
    // std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    // std::uniform_real_distribution<> dis(0.0, max_perturb);
    // return(dis(gen));

    ///// Gaussian
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0, stddev);

    return(dis(gen));
}

vector<vector<float>> get_noisy_pats(vector<vector<float>> pats,float stddev=0.0) {


    vector<vector<float>> noisypats(pats.size(), vector<float> (pats[0].size(),0));
    // noisypats.insert(noisypats.end(), pats.begin(), pats.end());
    for (int i = 0;i<pats.size();i++)
        for (int j = 0;j<pats[i].size();j++) {
            // if (pats[i][j]==1.0)
            //     noisypats[i][j] = pats[i][j] - getnoiseperturb(max_perturb);
            // else
                noisypats[i][j] = pats[i][j] + getnoiseperturb(stddev); 
                if (noisypats[i][j]<0)
                    noisypats[i][j] = 0; //pats[i][j] + abs(getnoiseperturb(stddev));

                // if (noisypats[i][j]>1)
                //     noisypats[i][j] = 1; //pats[i][j] + abs(getnoiseperturb(stddev));
                // if (noisypats[i][j]>1)
                //     noisypats[i][j]=1;
        }

    return(noisypats);

}
vector<vector<int>> getActiveUnits()
{
    /**
        Get active HCs or node index from file
    */
    string fname = "/home/rohan/Documents/Olfaction/Pattern_Generation/SNACKPatterns/cue_activeHCs.txt";

    vector <vector <int>> activeID;

    std::ifstream file(fname);
    std::string line;
    // std::getline(file, line);

    int p = 0;
    vector <int> tmpvec;
    while(getline(file, line))         //While you can read the file's content a line at a time, place its content into line
    {
        activeID.push_back(vector<int>()); // add row
        std::istringstream buffer(line);
        string num;
        while (getline(buffer, num, ' ')) {
            activeID.back().push_back(stoi(num)); // add number to last row
        } 

    }

    return activeID;
}

vector<float> getdefaultPat(vector<vector<float>> pats,int H,int M) {

    /*
        Create a default pat which is orthogonal to other pat in pats
    
    */


    vector<vector<int>> pat_MCs;
    vector<int> pushpat,defpat;

    ///// Get indices of MC in each pattern
    for (auto &pat:pats) {
        std::vector<float>::iterator it = pat.begin();
        while ((it = std::find_if(it, pat.end(), [](int x){return x > 0; })) != pat.end())
        {
            pushpat.push_back(std::distance(pat.begin(), it));
            it++;
        }
        std::for_each(pushpat.begin(), pushpat.end(), [M](int &c){ c = c%M;}); ////// Get the Unit within each Hypercolumn
        pat_MCs.push_back(pushpat);
        pushpat.clear();
    }

    // for(auto &i:pat_MCs) {
    //     for (auto &j:i)
    //         std::cout<<j<<" ";
    
    // std::cout<<std::endl;
    // }


    ///// Get unusued units in each HC and store as defpat

    vector<int> colvals,diff;
    vector<int> mcs(M);
    std::iota(mcs.begin(), mcs.end(), 0); //Fill with 0,1,2,....

    for (int col =0;col<pat_MCs[0].size();col++) {
        for (auto &p: pat_MCs)
            colvals.push_back(p[col]);
        std::sort(colvals.begin(),colvals.end());
        std::set_difference(mcs.begin(), mcs.end(), colvals.begin(), colvals.end(), std::back_inserter(diff));
        defpat.push_back(diff[0]);
        colvals.clear();
    }
    
    // for (auto i:defpat)
    //     std::cout<<i<<" ";


    //////// Convert defpat to binary pattern
    vector<float>defaultpat(H*M,0);
    int i = 0;
    for (auto unit: defpat) {
        defaultpat[i*M+unit] = 1;
        i++;
    }

    // for (auto i:defaultpat)
    //     std::cout<<i<<" ";


    return defaultpat;


}

vector<vector<float> > readJSONpats(string filename, string net_type) {

    std::ifstream file(filename);
    std::string str,substr;
    int nrow,HC,MC;
    if (net_type == "lang") {
        nrow = npat1;
        HC = H;
        MC = M;
    }
    else if (net_type == "od") {
        nrow = npat2;
        HC = H2;
        MC = M2;
    }

    int ncol = HC*MC;
    vector<vector<float> > pats(nrow,vector<float>(ncol,0));

    int p = 0, Hyp = 0, m=0,parse = 0;
    while (std::getline(file, str,'[')) {
        str.erase(std::remove(str.begin(), str.end(), ']'), str.end());
        std::istringstream s_stream(str);
        while(std::getline(s_stream,substr,','))
        {
           parse = 1;
           bool whiteSpacesOnly = substr.find_first_not_of (' ') == substr.npos;    //Gives 1 if substr is a blank space that I get at the end of string for some reason

           if(whiteSpacesOnly == 0) {
             m = stoi(substr);
             pats[p][Hyp*MC+m] = 1;
             Hyp++;
            }
        }
        if (parse == 1)
           p++;

        parse = 0;
        s_stream.clear();
        Hyp = 0;
    }

    if (net_type == "lang") {
        // npat1 = nrow;
        patsize1 = ncol;
    }
    else if (net_type == "od") {
        // npat2 = nrow;
        patsize2 = ncol;
    }
    else
        throw std::runtime_error("readJSONpats: Invalid net_type!");


    if (net_type == "lang")
         pats.assign(pats.begin(), pats.begin() + npat1);
    if (net_type == "od")
         pats.assign(pats.begin(), pats.begin() + npat2);

    // for (auto i:pats[0])
    //     std::cout<<i<<" ";
    // std::cout<<endl;

    return pats;

}

vector<float> partialize_pattern(vector<float> pat, int silentHCs,int totalHCs,int scalepat=1,int writepat = 1) {
    /*
        Given number of HCs to silence: silentHCs, randomly choose HCs in pattern and make them silent.

        Need to add some code here to save the active units to file to help with visualisation
    */



    if (silentHCs == 0) {
        return pat;
    }

    int activeHCs = totalHCs-silentHCs;
    vector<int> active_units(totalHCs,0);
    vector<int> cueUnits;
    vector<float> partial_pat(pat.size(),0);


    int counter = 0;


    for(int m = 0;m<pat.size();m++) {
        if (pat[m] == 1) {
            active_units[counter] = m;
            counter++;
        }
    }

    std::random_device rd;
    std::mt19937 g(rd());

    std::shuffle(std::begin(active_units),std::end(active_units),g);
    cueUnits = std::vector<int>(active_units.begin(), active_units.begin()+activeHCs);



    for (auto j: cueUnits) partial_pat[j]=scalepat*1.0;

    vector<float> active_units_float(active_units.begin(), active_units.end());

    if (writepat == 1)
        if (runflag=="encode_local_asso")
            trnphase_partialpats.push_back(partial_pat);
        else if (runflag=="preload_localasso")
            recallphase_partialpats.push_back(partial_pat);
        else if (runflag=="full")
            recallphase_partialpats.push_back(partial_pat);   


    return partial_pat;

        
}

vector<vector<float>> make_custom_partialcues(string partial_mode,string net,int HC,int MC,vector<int> intensity_based_units = {}) {
 
    /*
        Cue only some hypercolumns of each pattern using list HCs
        Creates partial patterns out of whole set of patterns. 
        For creating partial pattern individually/one at a time, see previous related function partialize_pattern()

        intensity_based_units is a vector that contains the number of units on top of cueHCs that need to be added for each
        odor based on k means clustering with intensity ratings.

    */

    vector<vector<float>> trpats;
    int patsize;
    if (net=="LTM1") {
        patsize = patsize1;
        trpats = trpats1;    
    }
    else if (net=="LTM2") {
        patsize = HC*MC;
        trpats = trpats2;
    }

    vector<vector<float>> partial_cues(trpats.size(), vector<float> (patsize,0));

    vector<vector<int>> active_units(trpats.size(),vector<int> (HC,0));


    if (partial_mode == "uniform") {

        
        vector<float> cues = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
        int unit;
        for (int i=0;i<trpats.size();i++) {

            for (int h=0;h<cueHCs;h++) {
                for(int m=0;m<M;m++) {
                    unit = h*M+m;
                    partial_cues[i][unit] = trpats[i][unit];
                }
            }
        }
    }


    else {
        ////// Cue HCs which share most overlap or least overlap over patterns
        int counter;
        for(int i = 0;i<partial_cues.size();i++) {
            counter = 0;
            for(int m = 0;m<trpats[i].size();m++)
                if (trpats[i][m] == 1) {
                    active_units[i][counter] = m;
                    counter++;
                }
        }

        if (partial_mode== "random") {

            std::random_device rd;
            std::mt19937 g(rd());
            vector<int> pat,cueUnits;
            for (int i = 0;i<active_units.size();i++) {
                pat = active_units[i];
                std::shuffle(std::begin(pat),std::end(pat),g);
                cueUnits = std::vector<int>(pat.begin(), pat.begin()+cueHCs+intensity_based_units[i]);
                // for (auto j:cueUnits)
                //     std::cout<<j<<" ";
                // std::cout<<std::endl;
                for (auto j: cueUnits) partial_cues[i][j]=1.0;
            }
            return partial_cues;

        }

        vector<vector<int>> overlaps(trpats.size(),vector<int> (HC,0));
        // vector<vector<int>> overlaps2(trpats2.size(),vector<int> (H2,0)); 
        int overlap,unit;

        for (int h = 0;h<HC;h++) {
            for (int pat = 0;pat<active_units.size();pat++) {
                unit = active_units[pat][h];
                overlap = 0;
                for(int pat2 = 0; pat2<active_units.size();pat2++) {
                    if (pat2==pat)
                        continue;
                    if (active_units[pat2][h]==unit)
                        overlap++;
                }
                overlaps[pat][h] = overlap;
               
            }
        }


        vector<vector<int>> top_overlaps(trpats.size(),vector<int> (cueHCs,0));
        // vector<vector<int>> top_overlaps2(trpats2.size(),vector<int> (cueHCs,0));
        vector <int> indices(HC);
        // vector <int> indices2(H2);
        vector <int> overlap_pat(overlaps[0].size());
        // vector <int> overlap_pat2(overlaps2[0].size());

        for (int pat = 0;pat<top_overlaps.size();pat++) {
            std::iota(indices.begin(), indices.end(), 0); //Fill with 0,1,2,....

            overlap_pat = overlaps[pat];
            if (partial_mode=="least_overlap")
                std::partial_sort(indices.begin(), indices.begin()+cueHCs, indices.end(),
                                [overlap_pat](int i,int j) {return overlap_pat[i]<overlap_pat[j];});   //Sort cueHC number of elements in overlap list
            else if (partial_mode =="most_overlap")    
                std::partial_sort(indices.begin(), indices.begin()+cueHCs, indices.end(),
                                [overlap_pat](int i,int j) {return overlap_pat[i]>overlap_pat[j];});   //Sort cueHC number of elements in overlap list   

            top_overlaps[pat] = vector<int>(indices.begin(), indices.begin()+cueHCs);

        }


        int idx;
        for(int pat = 0;pat<trpats.size();pat++) {
            for(int h=0;h<top_overlaps[pat].size();h++) {
                idx = top_overlaps[pat][h];
                unit = active_units[pat][idx];
                partial_cues[pat][unit]=1;
            }
        } 


        // for(int i = 0;i<partial_cues[1].size();i++) {
        //         std::cout<<partial_cues[1][i]<<" ";
        // }
        // std::cout<<std::endl;


    }

    return partial_cues;
}



vector<vector<float>> distort_pats(vector<vector<float>> pats,int HC,int MC,string savefname,int distort = 1,int silence = 0,int use_intensity_ratings = 0) {
    /*
        Randomly flip active unit in 'distort' number of HCs in pattern. Choice of random unit excludes units in HC that have never been active
    */

    vector<vector<float>> rp(pats.size(), vector<float>(HC*MC,0)); 
    vector<vector<int>> active_units(pats.size(),vector<int> (HC,0));
    vector<vector<int>> active_units_transpose(HC,vector<int> (pats.size(),0));
    vector<int> HCs(HC,0);


    vector<vector<float>> distortedHC_ids(pats.size());
    // vector<vector<float>> distortedHC_ids(pats.size(),vector<float>(distort));

    int distortHC_id;
    for (int i = 0;i<HC;i++)
        HCs[i] = i;
    

    vector <int> intensitybased_distort;
    if (use_intensity_ratings==1) {
        // NOTE: Assumes usage of odor patterns
        int max_HCs_distort = 8;
        vector <float> intensity_ = {0.19621622, 0.52162162, 0.39432432, 0.19567568, 0.38756757,
       0.32702703, 0.30945946, 0.2827027 , 0.22783784, 0.35810811,
       0.32972973, 0.23945946, 0.37621622, 0.38567568, 0.39135135,
       0.26108108}; //// 1-intensity ratings

        float max,min,intensity_diff,range_diff;
        max = intensity_[1];
        min = intensity_[3];
        intensity_diff = max-min;

        range_diff = max_HCs_distort - 0; 

        ///// Convert intensity ratings to number of columns to be distorted
        for (int i = 0;i<pats.size();i++) {
            intensitybased_distort.push_back(int(range_diff*(intensity_[i]-min)/(intensity_diff)));
        }

    }

    for (auto i: intensitybased_distort)
        std::cout<<i<<" ";

    int counter;
    //Get id of active units
    for(int i = 0;i<pats.size();i++) {
        counter = 0;
        for(int m = 0;m<pats[i].size();m++)
            if (pats[i][m] == 1) {
                active_units[i][counter] = m;
                counter++;
            }
    }

    //Transpose active_units so shape = HC x npats
    for (int i = 0;i<active_units[0].size();i++) {
        for(int j = 0;j<active_units.size();j++) {
            active_units_transpose[i][j] = active_units[j][i]; //std::cout<<vec[j][i]<<" ";
        }
        
    }

    int active,flipto_unit,distort_count;
    vector<int> units_list;
    for (int i = 0;i<pats.size();i++)
    {
        //Shuffle HCs
        std::random_device rd;
        std::mt19937 g(rd());  
        // std::mt19937 g(seed);
        std::shuffle(std::begin(HCs),std::end(HCs),g);


        rp[i] = pats[i];
        if(use_intensity_ratings==1) 
            distort_count = intensitybased_distort[i];
        else
            distort_count = distort;
        for(int id = 0;id<distort_count;id++) {
            //get id of HC to distort in pattern and store in distortHC_ids to write to file later
            distortHC_id = HCs[id];
            distortedHC_ids[i].push_back(float(distortHC_id));
            // std::cout<<"\ni: "<<i<<" distortHC: "<<distortHC_id<<std::endl;

            //Get active unit in that HC for that pattern
            active = active_units[i][distortHC_id];
            // std::cout<<"active: "<<active<<std::endl;

            // Get list of active units in that HC across patterns
            units_list = active_units_transpose[distortHC_id];

            //Extract unique units (remove re-ocurrences)
            sort(units_list.begin(),units_list.end());
            units_list.erase(unique(units_list.begin(),units_list.end()),units_list.end());

            //Remove active unit from units_list 
            units_list.erase(std::find(units_list.begin(),units_list.end(),active));

            //Randomly select on unit from the units_list
            // std::mt19937 g(rd());    // Random each time
            std::mt19937 g(seed);    // Seed dependent for easier visual comparison 
            std::uniform_int_distribution<int> dist(0, units_list.size() - 1);
            flipto_unit = units_list[dist(g)];

            //Set inp 0 for previously active unit, and inp 1 to new unit in HC (if silence = 0, we just silence the hypercolumn by setting all units to 0)
            if (silence == 1) {
                rp[i][active] = 0.0;
                rp[i][flipto_unit] = 0.0;
            }
            else {
            rp[i][active] = 0.0;
            rp[i][flipto_unit] = 1.0;
            }

            //std::shuffle(std::begin(rp[i])+(distortHC_id*MC),std::begin(rp[i])+(distortHC_id*MC+MC),g);
            // std::cout<<"Distorting pattern "<<i<<" HC "<<distortHC_id<<std::endl;
        }
        distortedHC_ids[i].push_back(-1);
    }


    fwritemat(distortedHC_ids,savefname);
    return rp;
}

std::map<int, vector<int> > getOdorwiseAssocs(){
/*
    Creates a map that stores odor ids as keys and vector of descriptor ids as values
*/


    vector <int> v;    
    std::map<int, vector<int> > m;
    for (int od=0;od<npat2;od++) {

        for (size_t i=0; i<patstat.size(); i++) {
            if (patstat[i][1]==od)
                v.push_back(patstat[i][0]);

        }
        m.insert({od, v});
        v.clear();
    }

    return(m);
}

std::map<int, int> getOdorwisetrnreps(vector<int> fam){
/*
    Creates a map that stores odor ids as keys and number of training reps as values
    training rep counts were obtained by k means clustering of familiarity ratings
*/

  
    std::map<int, int> m;
    for (int od=0;od<npat2;od++) {
        m.insert({od, fam[od]});
    }

    return(m);
}


void setupLTM(int argc,char **args,int single_sub_flag = 1) {

    ginitialize();


    string PATH = "/home/rohan/Documents/BCPNNSimv2/works/apps/olflang/";

    string paramfile = PATH+"olflangmain1.par";

    parseparams(paramfile);

    //If additional params given in terminal it should be in the following format: param name, param value, param name, param value....
    //This is used in OmmisionSensitivity_Hyperparams.py
    std::cout<<"Default Recuwgain: "<<recuwgain<<"  Assowgain: "<<assowgain<<"  Bgain: "<<bgain<<std::endl;
    if (argc>1) {

        for (int i = 1;i<argc;i+=2) {
            if (strcmp(args[i], "recuwgain") == 0)
                recuwgain = atof(args[i+1]);
            else if (strcmp(args[i], "assowgain") == 0)
                assowgain = atof(args[i+1]);
            else if (strcmp(args[i], "bgain") == 0)
                bgain = atof(args[i+1]);
            else
                throw std::runtime_error("setupLTM: Invalid command line argument");
        }

    }
    std::cout<<"Modified Recuwgain: "<<recuwgain<<"  Assowgain: "<<assowgain<<"  Bgain: "<<bgain<<std::endl;
    gsetseed(seed);

    // getpatstat("/home/rohan/Documents/BCPNNSimv2/works/apps/olflang/patstat_top3snackdescs.txt");
    // getpatstat("/home/rohan/Documents/BCPNNSimv2/works/apps/olflang/patstat_n3.txt");
    // getpatstat("/home/rohan/Documents/BCPNNSimv2/works/apps/olflang/patstat_topdescs3e-01thresh.txt");
    // getpatstat(PATH+"patstat_16od_16descs.txt");
    getpatstat(PATH+"patstat_si_nclusters4_topdescs.txt");
    // getpatstat(PATH+"patstat_correctdescs_maxassocs4.txt");

    // string binfname = "/home/rohan/Documents/BCPNNSimv2/works/apps/olflang/InpPats/artif_binpats_nonmodular_tester2(sparse)_npats5.txt";
    // string binfname = "/home/rohan/Documents/Olfaction/Pattern_Generation/SNACKPatterns/npats10_patsize303_random(modular)"; 
    // string binfname = "/home/rohan/Documents/Olfaction/Pattern_Generation/SNACKPatterns/meanpats16.txt";
    // trpats = getpats(binfname,single_sub_flag);

    std::cout<<"H1: "<<H<<" M1: "<<M<<" H2: "<<H2<<" M2: "<<M2<<" npat1: "<<npat1<<" npat2: "<<npat2<<" patstat.size "<<patstat.size()<<endl;

    // //Language Network
    // trpats1 = readJSONpats( "/home/rohan/Documents/Olfaction/Pattern_Generation/GMCPatterns/w2v_D0_npat16_emax06_H20_M20_v2.json","lang");  
    // trpats1 = readJSONpats( "/home/rohan/Documents/Olfaction/Pattern_Generation/GMCPatterns/w2v16_D0_npat16_emax06_H20_M20_v2_logTransformed.json","lang");   
    // trpats1 = readJSONpats( "/home/rohan/Documents/Olfaction/Pattern_Generation/GMCPatterns/w2v_top3descs_D0_npat35_emax06_H20_M20_v2_logTransformed.json","lang"); 
    // trpats1 = readJSONpats( "/home/rohan/Documents/Olfaction/Pattern_Generation/GMCPatterns/w2v_topdescs05thresh_D0_npat24_emax06_H20_M20_v2.json","lang");  
    // trpats1 = readJSONpats( "/home/rohan/Documents/Olfaction/Pattern_Generation/GMCPatterns/w2v_topdescs05thresh_D0_npat24_emax06_H20_M20_v2_sigmoidTransformed(E=5).json","lang"); 
    // trpats1 = readJSONpats( "/home/rohan/Documents/Olfaction/Pattern_Generation/GMCPatterns/w2v16_D0_npat16_emax06_H20_M20_v2_sigmoidTransformed(E=10).json","lang");
    // trpats1 = readJSONpats( "/home/rohan/Documents/Olfaction/Pattern_Generation/GMCPatterns/w2v_topdescs05thresh_D0_npat24_emax06_H20_M20_v2_sigmoidTransformed(beta=5 gamma=10).json","lang");
    // trpats1 = readJSONpats( "/home/rohan/Documents/Olfaction/Pattern_Generation/GMCPatterns/w2v_topdescs033thresh_D0_npat27_emax20_H20_M20_v2_test_Dscaling1_15.json","lang");
    
    // trpats1 = readJSONpats( "/home/rohan/Documents/Olfaction/Pattern_Generation/GMCPatterns/w2v_si_4clusters_D0_npat33_emax20_H20_M20_v2_test_Dscaling1_3.json","lang");
    

    // trpats1 = readJSONpats( "/home/rohan/Documents/Olfaction/Pattern_Generation/GMCPatterns/w2v_si_4clusters_D0_npat33_emax20_H20_M20_v3_Dscaling1_15.json","lang");
    // trpats1 = readJSONpats( "/home/rohan/Documents/Olfaction/Pattern_Generation/GMCPatterns/w2v_si_4clusters_D0_npat33_emax20_H20_M20_v3_Dscaling1_15_dampen_factor3.json","lang");
    // trpats1 = readJSONpats( "/home/rohan/Documents/Olfaction/Pattern_Generation/GMCPatterns/w2v_si_4clusters_npat33_emax20_H20_M20_dampen_factor3_5.json","lang");
  
    trpats1 = readJSONpats( "/home/rohan/Documents/Olfaction/Pattern_Generation/GMCPatterns/w2v_si_4clusters_npat33_emax20_H15_M15_dampen_factor3_5.json","lang");
    // trpats1 = readJSONpats( "/home/rohan/Documents/Olfaction/Pattern_Generation/GMCPatterns/w2v_si_4clusters_npat36_emax20_H15_M15_dampen_factor3_5_Age&MMSEFilteredSNACKData.json","lang");

    

    // trpats1 = readJSONpats( "/home/rohan/Documents/Olfaction/Pattern_Generation/GMCPatterns/w2v_si_2clusters_44_emax20_H15_M15_dampen_factor2.json","lang");
    // trpats1 = readJSONpats( "/home/rohan/Documents/Olfaction/Pattern_Generation/GMCPatterns/w2v_correctdescs_maxassocs4_npat51_emax20_H15_M15_dampen_factor4.json","lang");
    // trpats1 = readJSONpats( "/home/rohan/Documents/Olfaction/Pattern_Generation/GMCPatterns/w2v_si_4clusters_npat33_emax20_H15_M15_dampen_factor1_new.json","lang");
    
    
    // trpats1 = readJSONpats( "/home/rohan/Documents/Olfaction/Pattern_Generation/GMCPatterns/w2v_single_labels_npat16_emax20_H15_M15_dampen_factor2_new.json","lang");
    
    // //Odor Network

    // trpats2 = readJSONpats( "/home/rohan/Documents/Olfaction/Pattern_Generation/GMCPatterns/odors_D0_npat16_emax06_H20_M20_v2.json","od");
    // trpats2 = readJSONpats( "/home/rohan/Documents/Olfaction/Pattern_Generation/RobPats/patterns_v3(snack_sorted).json","od");
    //trpats2 = readJSONpats( "/home/rohan/Documents/Olfaction/Pattern_Generation/GMCPatterns/odors_D0_npat16_emax06_H20_M20_v2_logTransformed.json","od");
    // trpats2 = readJSONpats( "/home/rohan/Documents/Olfaction/Pattern_Generation/GMCPatterns/odors_D0_npat16_emax06_H20_M20_v2_sigmoidTransformed(E=10).json","od");
    // trpats2 = readJSONpats( "/home/rohan/Documents/Olfaction/Pattern_Generation/GMCPatterns/odors_D0_npat16_emax06_H20_M20_v2_sigmoidTransformed(E=5 m=04).json","od");
    // trpats2 = readJSONpats( "/home/rohan/Documents/Olfaction/Pattern_Generation/GMCPatterns/odors_D0_npat16_emax06_H20_M20_v2_sigmoidTransformed(beta=5 gamma=11).json","od");
    
    // trpats2 = readJSONpats( "/home/rohan/Documents/Olfaction/Pattern_Generation/GMCPatterns/odors_D0_npat16_emax20_H20_M20_v2_test_Dscaling1_4.json","od");
    // trpats2 = readJSONpats( "/home/rohan/Documents/Olfaction/Pattern_Generation/GMCPatterns/odors_D0_npat16_emax20_H20_M20_v3_Dscaling1_2.json","od");
    // trpats2 = readJSONpats( "/home/rohan/Documents/Olfaction/Pattern_Generation/GMCPatterns/odors_D0_npat16_emax20_H20_M20_v3_Dscaling1_15_dampen_factor3.json","od");
    // trpats2 = readJSONpats( "/home/rohan/Documents/Olfaction/Pattern_Generation/GMCPatterns/odors_npat16_emax20_H20_M20_dampen_factor3_5.json","od");
    
    trpats2 = readJSONpats( "/home/rohan/Documents/Olfaction/Pattern_Generation/GMCPatterns/odors_npat16_emax20_H15_M15_dampen_factor3_5.json","od");

    // trpats2 = readbinpats("/home/rohan/Documents/Olfaction/Pattern_Generation/graded_intensitybased_odpats.bin",16,400,"od");
    // trpats2 = readbinpats("/home/rohan/Documents/Olfaction/Pattern_Generation/uniformgraded_intensitybased_odpats.bin",16,400,"od");
    // trpats2 = readbinpats("/home/rohan/Documents/Olfaction/Pattern_Generation/nonnormalized_intensitybased_odpats.bin",16,400,"od");
    
    /////ORTHO PATTERNS 
    // npat1 = 33;
    // trpats1 = mkpats(npat1,H,M,ORTHO); 
    // trpats2 = mkpats(npat2,H2,M2,ORTHO);




    /////// Read odor intensity silent HCs file
    if (use_intensity==1)
        init_intensityHCsmap(PATH+"od_intensity_noise_20HCs_6e-01max_noise.txt");

    // printf("SUBJECTS, ODORS, PATSIZE: %d %d %d \n",subjects,odors,patsize);
    if (epsfloat_multiplier==0)
        error("olflangmain1::setupLTM","cannot epsfloat_multiplier to 0");

    else {
        epsfloat *= epsfloat_multiplier;
        if (epsfloat_multiplier!=1)     std::cout<<"\nEPSFLOAT: "<<epsfloat<<endl;

    }


    vector<float> distance = {assoc_dist,0,0};

    ltm1 = new PopH(H,M,BCP,HALF);
    ltm1->setgeom(HEX2D);
    ltm1->fwritegeom("geom1.bin");

    ltm1->setseed(seed);
    ltm1->setparam("igain",igain);
    ltm1->setparam("again",again);
    ltm1->setparam("taum",taum);
    ltm1->setparam("taua",taua);
    ltm1->setparam("nmean",nmean);
    ltm1->setparam("namp",namp);

    prj11 = new PrjH(ltm1,ltm1,pdens,INCR);

    prj11->setparam("wdens",wdens);
    prj11->setparam("taup",taup);
    prj11->setparam("taucond",taucond);
    prj11->setparam("cspeed",cspeed);
    prj11->setparam("bgain",bgain);


    ltm2 = new PopH(H2,M2,BCP,HALF);
    ltm2->setgeom(HEX2D,distance);
    ltm2->fwritegeom("geom2.bin");

    ltm2->setseed(seed);
    ltm2->setparam("igain",igain);
    ltm2->setparam("again",again);
    ltm2->setparam("taum",taum);
    ltm2->setparam("taua",taua);
    ltm2->setparam("nmean",nmean);
    ltm2->setparam("namp",namp);

    prj22 = new PrjH(ltm2,ltm2,pdens,INCR);

    prj22->setparam("wdens",wdens);
    prj22->setparam("taup",taup);
    prj22->setparam("taucond",taucond);
    prj22->setparam("cspeed",cspeed);
    prj22->setparam("bgain",bgain);


    prj12 = new PrjH(ltm1,ltm2,assocpdens,INCR);

    prj12->setparam("wdens",wdens);
    prj12->setparam("taup",taup);
    prj12->setparam("taucond",taucond);
    prj12->setparam("cspeed",assoc_cspeed);
    prj12->setparam("bgain",bgain);


    prj21 = new PrjH(ltm2,ltm1,assocpdens,INCR);

    prj21->setparam("wdens",wdens);
    prj21->setparam("taup",taup);
    prj21->setparam("taucond",taucond);
    prj21->setparam("cspeed",assoc_cspeed);
    prj21->setparam("bgain",bgain);


    prj11->setparam("tausd",tausd);
    prj12->setparam("tausd",tausd);
    prj21->setparam("tausd",tausd);
    prj22->setparam("tausd",tausd);
    
    prj11->setparam("tausf",tausf);
    prj12->setparam("tausf",tausf);
    prj21->setparam("tausf",tausf);
    prj22->setparam("tausf",tausf);
    
    prj11->setparam("p0",p0);
    prj12->setparam("p0",p0);
    prj21->setparam("p0",p0);
    prj22->setparam("p0",p0);



    setuplogging();


}
void preloadBW(string mode) {
    // To preload stored weights and biases from .bin files instead of running encoding phase
    vector<vector<float> > Wij11, Wij22, Wij12,Wij21;
    vector<float> Bj11, Bj22, Bj12, Bj21 ;
    int N1 = H*M;
    int N2 = H2*M2;
    // Wij11 = readbinmat("Wij11pre16.bin",H*M);
    // Wij22 = readbinmat("Wij22pre16.bin",H2*M2);
    // Wij12 = readbinmat("Wij12pre16.bin",H*M);
    // Wij21 = readbinmat("Wij21pre16.bin",H2*M2);
    // Bj11  = readbinvec("Bj11pre16.bin",H*M);
    // Bj22  = readbinvec("Bj22pre16.bin",H2*M2);
    // Bj21  = readbinvec("Bj21pre16.bin",H*M);
    // Bj12  = readbinvec("Bj12pre16.bin",H2*M2);

    if (use_familiarity==1) {
    // Wij11 = readbinmat("Wij11pre_si_2clusters_withFam.bin",H*M);
    // Wij22 = readbinmat("Wij22pre_si_2clusters_withFam.bin",H2*M2);
    // Wij12 = readbinmat("Wij12pre_si_2clusters_withFam.bin",H*M);
    // Wij21 = readbinmat("Wij21pre_si_2clusters_withFam.bin",H2*M2);
    // Bj11  = readbinvec("Bj11pre_si_2clusters_withFam.bin",H*M);
    // Bj22  = readbinvec("Bj22pre_si_2clusters_withFam.bin",H2*M2);
    // Bj21  = readbinvec("Bj21pre_si_2clusters_withFam.bin",H*M);
    // Bj12  = readbinvec("Bj12pre_si_2clusters_withFam.bin",H2*M2);
    }
    else {

    // Wij11 = readbinmat("Wij11pre_si_4clusters_LangNetDF3,5_OdNetDF3,5.bin",N1,N1);
    // Wij22 = readbinmat("Wij22pre_si_4clusters_LangNetDF3,5_OdNetDF3,5.bin",N2,N2);
    // Wij12 = readbinmat("Wij12pre_si_4clusters_LangNetDF3,5_OdNetDF3,5.bin",N1,N2);
    // Wij21 = readbinmat("Wij21pre_si_4clusters_LangNetDF3,5_OdNetDF3,5.bin",N2,N1);
    // Bj11  = readbinvec("Bj11pre_si_4clusters_LangNetDF3,5_OdNetDF3,5.bin",N1);
    // Bj22  = readbinvec("Bj22pre_si_4clusters_LangNetDF3,5_OdNetDF3,5.bin",N2);
    // Bj21  = readbinvec("Bj21pre_si_4clusters_LangNetDF3,5_OdNetDF3,5.bin",N1);
    // Bj12  = readbinvec("Bj12pre_si_4clusters_LangNetDF3,5_OdNetDF3,5.bin",N2);

    // Wij11 = readbinmat("Wij11pre_si_4clusters_Ortho.bin",N1,N1);
    // Wij22 = readbinmat("Wij22pre_si_4clusters_Ortho.bin",N2,N2);
    // Wij12 = readbinmat("Wij12pre_si_4clusters_Ortho.bin",N1,N2);
    // Wij21 = readbinmat("Wij21pre_si_4clusters_Ortho.bin",N2,N1);
    // Bj11  = readbinvec("Bj11pre_si_4clusters_Ortho.bin",N1);
    // Bj22  = readbinvec("Bj22pre_si_4clusters_Ortho.bin",N2);
    // Bj21  = readbinvec("Bj21pre_si_4clusters_Ortho.bin",N1);
    // Bj12  = readbinvec("Bj12pre_si_4clusters_Ortho.bin",N2);

    Wij11 = readbinmat("Wij11pre_si_4clusters_GasolineSingleAssoc.bin",N1,N1);
    Wij22 = readbinmat("Wij22pre_si_4clusters_GasolineSingleAssoc.bin",N2,N2);
    Wij12 = readbinmat("Wij12pre_si_4clusters_GasolineSingleAssoc.bin",N1,N2);
    Wij21 = readbinmat("Wij21pre_si_4clusters_GasolineSingleAssoc.bin",N2,N1);
    Bj11  = readbinvec("Bj11pre_si_4clusters_GasolineSingleAssoc.bin",N1);
    Bj22  = readbinvec("Bj22pre_si_4clusters_GasolineSingleAssoc.bin",N2);
    Bj21  = readbinvec("Bj21pre_si_4clusters_GasolineSingleAssoc.bin",N1);
    Bj12  = readbinvec("Bj12pre_si_4clusters_GasolineSingleAssoc.bin",N2);

    // Wij11 = readbinmat("Wij11pre_si_4clusters_15x15.bin",N1,N1);
    // Wij22 = readbinmat("Wij22pre_si_4clusters_15x15.bin",N2,N2);
    // Wij12 = readbinmat("Wij12pre_si_4clusters_15x15.bin",N1,N2);
    // Wij21 = readbinmat("Wij21pre_si_4clusters_15x15.bin",N2,N1);
    // Bj11  = readbinvec("Bj11pre_si_4clusters_15x15.bin",N1);
    // Bj22  = readbinvec("Bj22pre_si_4clusters_15x15.bin",N2);
    // Bj21  = readbinvec("Bj21pre_si_4clusters_15x15.bin",N1);
    // Bj12  = readbinvec("Bj12pre_si_4clusters_15x15.bin",N2);
        
    // Wij11 = readbinmat("Wij11pre_correctdescs_maxassocs4.bin",H*M);
    // Wij22 = readbinmat("Wij22pre_correctdescs_maxassocs4.bin",H2*M2);
    // Wij12 = readbinmat("Wij12pre_correctdescs_maxassocs4.bin",H*M);
    // Wij21 = readbinmat("Wij21pre_correctdescs_maxassocs4.bin",H2*M2);
    // Bj11  = readbinvec("Bj11pre_correctdescs_maxassocs4.bin",H*M);
    // Bj22  = readbinvec("Bj22pre_correctdescs_maxassocs4.bin",H2*M2);
    // Bj21  = readbinvec("Bj21pre_correctdescs_maxassocs4.bin",H*M);
    // Bj12  = readbinvec("Bj12pre_correctdescs_maxassocs4.bin",H2*M2);
    }
 

    ////// Set Uniform Bias
    // float m = calculateMean(Bj11);
    // std::cout<<"Bj11 mean: "<<m<<std::endl;
    // Bj11.assign(N1, m);
    // m = calculateMean(Bj21);
    // Bj21.assign(N1, m);
    // std::cout<<"Bj21 mean: "<<m<<std::endl; //NOTE TO SELF: This WORKS
 

    // float sum;
    // sum = std::accumulate(std::begin(Bj11), std::end(Bj11), 0.0);
    // Bj11.assign(H*M, sum/Bj11.size());

    // sum = std::accumulate(std::begin(Bj22), std::end(Bj22), 0.0);
    // Bj22.assign(H2*M2, sum/Bj22.size());

    // sum = std::accumulate(std::begin(Bj12), std::end(Bj12), 0.0);
    // Bj12.assign(H2*M2, sum/Bj12.size());

    // sum = std::accumulate(std::begin(Bj21), std::end(Bj21), 0.0);
    // Bj21.assign(H*M, sum/Bj21.size());


    if (mode == "local") {
    setmode("encoderecu","recu");

    printf("Loading local weights for ltm1 and ltm2 (%d)\n",simstep);
    initPWB("prj11",Wij11,Bj11,H,M);
    initPWB("prj22",Wij22,Bj22,H2,M2);

    resetstate();
    }
    else if (mode == "localasso") {

    setmode("encode","alla");

    printf("Loading local and asso weights for ltm1 and ltm2 (%d)\n",simstep);
    initPWB("prj11",Wij11,Bj11,N1,N1);
    initPWB("prj22",Wij22,Bj22,N2,N2);
    initPWB("prj12",Wij12,Bj12,N1,N2);
    initPWB("prj21",Wij21,Bj21,N2,N1);
    // initPWB("prj11",Wij11,H,M);
    // initPWB("prj22",Wij22,H2,M2);
    // initPWB("prj21",Wij21,H2,M2);
    // initPWB("prj12",Wij12,H,M);
    resetstate();
    }
}

void trainRecu(int init_run = 0) {



    // ltm1->setparam(ADGAIN,adgain); ltm2->setparam(ADGAIN,adgain);

    setmode("encoderecu","zero");

    // ltm1->setparam(NMEAN,nmean); ltm2->setparam(NMEAN,nmean);
    // printf("Training ltm1 (lang) & ltm2 (odor) (%d)\n",simstep);

    if (init_run==1) {
    // initial run to stabilize weights and biasese?
    ltm1->setparam(IGAIN,igain); ltm2->setparam(IGAIN,igain);
    ltm1->setinp(0); ltm2->setinp(0);
    resetstate();
    simulate(nstep*2);
    }

    std::random_device rd;
    std::mt19937 g(rd());

    vector<int> ltm1_items(npat1);
    std::iota (std::begin(ltm1_items), std::end(ltm1_items), 0);

    printf("Training ltm1 (lang) (%d)\n",simstep);
    ltm1->setparam(IGAIN,igain); ltm2->setparam(IGAIN,0);
    for(int rep = 0; rep<etrnrep; rep++) {

        if (encode_order=="random")
            std::shuffle(std::begin(ltm1_items),std::end(ltm1_items),g); 

        for(int p = 0; p<npat1;p++) {

            // std::cout<<"Epoch: "<<rep<<" LTM1 Item: "<<ltm1_items[p]<<std::endl;
            resetstate();
            prj11->setparam(PRN,1);
            ltm1->setinp(trpats1[ltm1_items[p]]);
            ltm2->setinp(0);
            simulate(nstep);
            simstages.push_back(simstep);

            ltm1->setinp(0);
            setPRN(0);
            simulate(ngap);

            simstages.push_back(simstep);
        }

    }

    printf("Training ltm2 (od) (%d)\n",simstep);


    vector<int> ltm2_items;
    if (use_familiarity==0) {
        ltm2_items.resize(npat2);
        std::iota (std::begin(ltm2_items), std::end(ltm2_items), 0);
    }
    else {
        vector <int> fam = {4,1,3,4,4,3,1,2,3,3,2,3,2,3,3,2};  //// [3,1,2,3,3,2,1,2,2,3,2,2,2,2,3,2]

        std::map<int, int> rep_counts = getOdorwisetrnreps(fam);
        for (int i =0;i<npat2;i++) {
            for(int j = 0;j<rep_counts.at(i);j++)
                ltm2_items.push_back(i);
        }
        
        for(auto i: ltm2_items)
            std::cout<<i<<" ";
        std::cout<<std::endl;
    }

    ltm1->setparam(IGAIN,0); ltm2->setparam(IGAIN,igain);
    for(int rep = 0; rep<etrnrep; rep++) {

        if (encode_order=="random")
            std::shuffle(std::begin(ltm2_items),std::end(ltm2_items),g); 

        for(int p = 0; p<ltm2_items.size();p++) {


            // std::cout<<"Epoch: "<<rep<<" LTM2 Item: "<<ltm2_items[p]<<std::endl;
            resetstate();
            prj22->setparam(PRN,1);
            ltm2->setinp(trpats2[ltm2_items[p]]);
            ltm1->setinp(0);
            simulate(nstep);
            simstages.push_back(simstep);

            ltm2->setinp(0);
            setPRN(0);
            simulate(ngap);

            simstages.push_back(simstep);
        }

        // if (rep+1<etrnrep)
        //     simstages.push_back(rep+1);
    }

    if (runflag=="encode_local_only")
        prj11->setparam(WGAIN,recuwgain); prj22->setparam(WGAIN,recuwgain);
    simulate(ngap*2);
    // simstages.push_back(-1);

}

void trainAssoc() {



    printf("Training ltm1<-->ltm2 association (%d)\n",simstep);
    setmode("encodeasso","zero");
    simstages.push_back(-1);
    simstages.push_back(simstep);
    std::random_device rd;
    std::mt19937 g(rd());
    vector<vector <int>> trained_sequences,encoded_odors; // keep track of sequences that are encoded each epoch
    // ltm1->setparam(NMEAN,nmean); ltm2->setparam(NMEAN,nmean);
    ltm1->setparam(IGAIN,igain); ltm2->setparam(IGAIN,igain);
    vector<int>seq;
    
    int item1,item2,silentHCs;
    int encode_flag = 0;

    // for (int i = 0;i<npat2;i++) {
    //     encoded_odors.push_back({i,0});
    //   }


    // for(int rep = 0; rep<atrnrep; rep++) {

    //   if (encode_order=="random")
    //     std::shuffle(std::begin(patstat),std::end(patstat),g);        

    //   ///// Reset encoded odors
    //   for (int i = 0;i<npat2;i++) {
    //     encoded_odors[i][1]=0;
    //   }

    //   for(int p = 0; p<patstat.size();p++)
    //   {

    //     resetstate();
       
    //     /// If odor is associated to only one descriptor
    //     /// Note: If you want to train all associations in every epoch instead of spreading it out, change == to >=
    //     if (ltm2_items_counts[patstat[p][1]][1]==1 || atrnrep==1) {
    //         item1 = patstat[p][0];
    //         item2 = patstat[p][1];
            
    //     }
    //     else {
    //         // // Odor is associated to multiple descs
    //         seq = {patstat[p][0],patstat[p][1]};
    //         if ((std::find(trained_sequences.begin(), trained_sequences.end(), seq) == trained_sequences.end()) && (encoded_odors[patstat[p][1]][1] == 0))  
    //         {
    //          ///// If sequence not in trained_sequences and odor has not been shown before in this epoch
    //             item1 = patstat[p][0];
    //             item2 = patstat[p][1];

    //             ///// if odor has not been encoded before in this epoch, add to encode_flag


    //             encoded_odors[patstat[p][1]][1] = 1;


    //             trained_sequences.push_back(seq);
                
    //         }
    //         else 
    //             continue;
    //    }


    //    std::cout<<"Epoch: "<<rep<<" Item1: "<<item1<<" Item2: "<<item2<<std::endl;

    //     //////// if patstat contains additional column referring to prn values, then use that,
    //     //////// else use 1 for all
    //     if (patstat[p].size()==3 && use_trneffort==1) {
    //         if (patstat[p][2]==2) {
    //             prj11->setparam(PRN,0);
    //             prj12->setparam(PRN,2); 
    //             prj21->setparam(PRN,2);
    //             prj22->setparam(PRN,0);
    //         }
    //         else {
    //             prj11->setparam(PRN,0);
    //             prj12->setparam(PRN,patstat[p][2]); 
    //             prj21->setparam(PRN,patstat[p][2]);
    //             prj22->setparam(PRN,0);
    //         }
    //     }
    //     else {
    //             prj12->setparam(PRN,1); 
    //             prj21->setparam(PRN,1);
    //     }
        
    //     //// In case we are using partial patterns for training based on odor intensity, randomly silence units
    //     //// in LTM2 pattern to be trained.

        
    //     // std::cout<<"Epoch: "<<rep+1<<" LTM1 Item: "<<item1<<" LTM2 Item: "<<item2<<std::endl;

    //     ltm1->setinp(trpats1[item1]);
        
    //     if (use_intensity==1) {
    //         silentHCs = get_odintensityHCs(item2);
    //         ltm2->setinp(partialize_pattern(trpats2[item2],silentHCs,H2,1,0));
    //     }
    //     else
    //         ltm2->setinp(trpats2[item2]);

    //     simulate(nstep);
    //     simstages.push_back(simstep);
    //     ltm1->setinp(0); ltm2->setinp(0);
    //     setPRN(0);
    //     simulate(ngap);
    //     simstages.push_back(simstep);
    //  }


    // }


    //////////////////////// Training paradigm with imbalances but where each odor is trained with one desc in an epoch
    int prn = 1,count=0,od;
    vector<int>odors(npat2),v;
    vector<vector<int>> trained_assoc_counts;
    for (int i = 0;i<npat2;i++) {
        trained_assoc_counts.push_back({i,0});
      }


    std::map<int, vector<int> > assocs = getOdorwiseAssocs();
    // for(auto i: assocs.at(2))
    //     std::cout<<"Assocs 0: "<<i<<"    "<<std::endl;
    std::iota(odors.begin(), odors.end(), 0);

    for (int rep=0;rep<atrnrep;rep++) {
        ////// For each epoch
        if (encode_order=="random")
            std::shuffle(std::begin(odors),std::end(odors),g);   
        // std::cout<<"\n";
        for (int i=0;i<npat2;i++) {
            ////// For each odor
            resetstate();
            od = odors[i];
            item2 = od;
            for(int j = 0;j<patstat.size();j++) {
                ///////For every row in patstat 
                if (patstat[j][1] == od) {
                    seq = {patstat[j][0],patstat[j][1]};

                    //// If association has not been trained before 
                    if (std::find(trained_sequences.begin(), trained_sequences.end(), seq) == trained_sequences.end())  {
                        trained_assoc_counts[od][1] +=1;
                        item1 = patstat[j][0]; 
                        trained_sequences.push_back(seq);
                        if (patstat[j].size()==3 && use_trneffort==1)
                            prn = patstat[j][2];
                        break;
                    }

                    //// If flow reaches here, then for the given odor, the association has not been trained
                    //// If every row in patstat file is a unique pair, then this should be reached when there are no longer any untrained assocs

                    v = assocs.at(od);
                    // for (auto z: v)
                    //     std::cout<<"v["<<od<<"]: "<<z<<std::endl;
                    if (trained_assoc_counts[od][1]==v.size()) {



                        if (patstat[j].size()==3 && use_trneffort==1)
                            prn = patstat[j][2];      

                        
                        for(int k=0;k<v.size();k++) {
                                // if(k==patstat[j][0])
                                //     continue;
                                seq = {v[k],od};
                            for (int l=0;l<trained_sequences.size();l++) {
                                if (trained_sequences[l][0]==seq[0] && trained_sequences[l][1]==seq[1]) {
                                    count++;
                                    // std::cout<<patstat[j][0]<<" "<<seq[0]<<" "<<seq[1]<<std::endl;
                                    trained_sequences.erase(std::next( std::begin( trained_sequences ), l ));
                                    break;
                                }
                            }

                        }
                        item1 = patstat[j][0];
                        trained_sequences.push_back({item1,od});
                        trained_assoc_counts[od][1] = 1;
                        count=0;
                        break;
                        // for(auto i:trained_sequences)
                        //     std::cout<<"trained_sequences: "<<i[0]<<"  "<<i[1]<<std::endl;
                    }


                }

            }

            prj12->setparam(PRN,prn); 
            prj21->setparam(PRN,prn);
            // std::cout<<"Epoch: "<<rep+1<<" LTM1 Item: "<<item1<<" LTM2 Item: "<<item2<<std::endl;
            ltm1->setinp(trpats1[item1]);
            ltm2->setinp(trpats2[item2]);
            simulate(nstep);
            simstages.push_back(simstep);
            ltm1->setinp(0); ltm2->setinp(0);
            setPRN(0);
            simulate(ngap);
            simstages.push_back(simstep);

        }

    }


    /////////////////////////// USING DEFAULT PATTERNS TO CORRECT IMBALANCES
    
    ///// Create dummy default pat for lang network
    // vector<float> defaultpat;
    // defaultpat = getdefaultPat(trpats1,H,M);
    // vector<int>odors(npat2);
    // vector<vector<int>> patstat2;
    // patstat2 = patstat;
    // int od,prn;
    // prn=1;
    // std::iota(odors.begin(), odors.end(), 0);

    // if (encode_order=="random")
    //     std::shuffle(std::begin(patstat2),std::end(patstat2),g);   

    // for (int rep=0;rep<atrnrep;rep++) {
    //     if (encode_order=="random")
    //         std::shuffle(std::begin(odors),std::end(odors),g);   

    //     for (int i=0;i<npat2;i++) {

    //         resetstate();
    //         od = odors[i];
    //         item1 = -1; //Default pat
    //         item2 = od;
    //         for(int j = 0;j<patstat2.size();j++) {

    //             if (patstat2[j][1] == od) {
    //                 item1 = patstat2[j][0];
    //                 if (patstat[j].size()==3 && use_trneffort==1)
    //                     prn = patstat[j][2];
    //                 //Remove association from patstat2
    //                 patstat2.erase(std::next( std::begin( patstat2 ), j )); 
    //                 break;
    //             }
    //         }


    //         prj12->setparam(PRN,prn); 
    //         prj21->setparam(PRN,prn);

    //         if (item1 == -1)
    //             ltm1->setinp(defaultpat);
    //         else
    //             ltm1->setinp(trpats1[item1]);

    //         std::cout<<"Epoch: "<<rep+1<<" LTM1 Item: "<<item1<<" LTM2 Item: "<<item2<<std::endl;

    //         ltm2->setinp(trpats2[item2]);
    //         simulate(nstep);
    //         simstages.push_back(simstep);
    //         ltm1->setinp(0); ltm2->setinp(0);
    //         setPRN(0);
    //         simulate(ngap);
    //         simstages.push_back(simstep);

    //     } 
    // }


    prj11->setparam(WGAIN,recuwgain); prj22->setparam(WGAIN,recuwgain);
    prj21->setparam(WGAIN,assowgain); prj12->setparam(WGAIN,assowgain);
    simulate(ngap*2);

}

void train_all_prjs() {
    // // // Train local and asso projections together


    printf("Training ltm1<-->ltm2 local & assoc projections (%d)\n",simstep);
    setmode("encode","zero");
    simstages.push_back(-1);
    simstages.push_back(simstep);
    std::random_device rd;
    std::mt19937 g(rd());
    vector<vector <int>> trained_sequences,encoded_odors; // keep track of sequences that are encoded each epoch
    vector<int>seq;
    ltm1->setparam(IGAIN,igain); ltm2->setparam(IGAIN,igain);


    for (int i = 0;i<npat2;i++) {
        encoded_odors.push_back({i,0});
    }
    int item1,item2,silentHCs;
    int encode_flag = 0;
    

    for(int rep = 0; rep<atrnrep; rep++) {

      if (encode_order=="random")
        std::shuffle(std::begin(patstat),std::end(patstat),g);        

      for(int p = 0; p<patstat.size();p++)
      {

        resetstate();
       
        /// If odor is associated to only one descriptor
        /// Note: If you want to train all associations in every epoch instead of spreading it out, change == to >=
        if (ltm2_items_counts[patstat[p][1]][1]==1) {
            item1 = patstat[p][0];
            item2 = patstat[p][1];
            
        }
        else {
            // // Odor is associated to multiple descs
            seq = {patstat[p][0],patstat[p][1]};
            if ((std::find(trained_sequences.begin(), trained_sequences.end(), seq) == trained_sequences.end()) && (encoded_odors[patstat[p][1]][1] == 0))  
            {
             ///// If sequence not in trained_sequences and odor has not been shown before in this epoch
                item1 = patstat[p][0];
                item2 = patstat[p][1];

                ///// if odor has not been encoded before in this epoch, add to encode_flag

                if (rep<etrnrep-1)
                    encoded_odors[patstat[p][1]][1] = 1;


                trained_sequences.push_back(seq);
                
            }
            else
                continue;
       }


       // std::cout<<"Epoch: "<<rep<<" Item1: "<<item1<<" Item2: "<<item2<<" trained_sequences size: "<<trained_sequences.size()<<std::endl;

        //////// if patstat contains additional column referring to prn values, then use that,
        //////// else use 1 for all
        if (patstat[p].size()==3 && use_trneffort==1) {
            if (patstat[p][2]==2) {
                prj11->setparam(PRN,1);
                prj12->setparam(PRN,2); 
                prj21->setparam(PRN,2);
                prj22->setparam(PRN,1);
            }
            else {
                prj11->setparam(PRN,1);
                prj12->setparam(PRN,patstat[p][2]); 
                prj21->setparam(PRN,patstat[p][2]);
                prj22->setparam(PRN,1);
            }
        }
        else
            setPRN(1);
        
        //// In case we are using partial patterns for training based on odor intensity, randomly silence units
        //// in LTM2 pattern to be trained.

        
        // std::cout<<"Epoch: "<<rep+1<<" LTM1 Item: "<<item1<<" LTM2 Item: "<<item2<<std::endl;

        ltm1->setinp(trpats1[item1]);
        
        if (use_intensity==1) {
            silentHCs = get_odintensityHCs(item2);
            ltm2->setinp(partialize_pattern(trpats2[item2],silentHCs,H2,1,0));
        }
        else
            ltm2->setinp(trpats2[item2]);

        simulate(nstep);
        simstages.push_back(simstep);
        ltm1->setinp(0); ltm2->setinp(0);
        setPRN(0);
        simulate(ngap);
        simstages.push_back(simstep);
     }

     encoded_odors.clear();
     for (int i = 0;i<npat2;i++) {
        encoded_odors.push_back({i,0});
     }
        // if (rep+1<etrnrep)
        //     simstages.push_back(rep+1);
    }

    // fwritemat(trnphase_partialpats,"trnphase_partialpats.bin");
    setPRN(0);
    prj11->setparam(WGAIN,recuwgain); prj22->setparam(WGAIN,recuwgain);
    prj21->setparam(WGAIN,assowgain); prj12->setparam(WGAIN,assowgain);
    simulate(ngap*2);

}


vector<int> findNonZeroIndices(vector<float>& pat) {
    vector<int> nonZeroIndices;
    for (int i = 0; i < pat.size(); i++) {
        if (pat[i] != 0) {
            nonZeroIndices.push_back(i);
        }
    }
    return nonZeroIndices;
}

vector<vector<float>> modify_od2lang_weights(vector<vector<float>> &w21, int nassocs,std::map<int, vector<int> >assocs,float reduction_factor) {
    /**
        Function to return a modified weight matrix given number of od->lang assocs, a map showing the assocs from each odor and
        the factor by which to modify with. The weights are divided by this factor (hence the name reduction factor for factor>1)
     **/
    vector<vector<float>> ret_w = w21;
    vector<int> v, odpat_indices, langpat_indices;
    //Loop through each odor
    for (int i = 0;i<npat2;i++) {

        //If number of descs associated to odor i == nassocs
        v = assocs.at(i);
        if(v.size()==nassocs) {
            //Get indices of odor pattern
            odpat_indices = findNonZeroIndices(trpats2[i]);
            //Loop through each associated descriptor and get the indices of language pattern
            for (auto desc: v) {
                langpat_indices = findNonZeroIndices(trpats1[desc]);
                //Modify the weights between each unit in odor pat and each unit in lang pat
                for (auto od_unit:odpat_indices)
                    for (auto lang_unit:langpat_indices)
                        ret_w[od_unit][lang_unit] /= reduction_factor;
                
            }
        }
        else
            continue;
    }

    return ret_w;
}
void recall2nets() {

    vector<string> ODORS = {"Gasoline", "Leather", "Cinnamon", "Pepparmint","Banana", "Lemon", "Licorice", "Terpentine",
            "Garlic", "Coffee", "Apple", "Clove","Pineapple", "Rose", "Mushroom", "Fish"};
    vector<float> cues = {0};
    // vector<int> intensity_based_units = {4, 0, 2, 4, 2, 2, 3, 3, 4, 2, 2, 3, 2, 2, 2, 3};
    vector<int> intensity_based_units(16,0);
    // vector<float> igains = {10.,  1.,  5., 10.,  5.,  6.,  7.,  8.,  9.,  6.,  6.,  9.,  5., 5.,  5.,  8.};
    // vector<float> cues = {3,3,3,3,3,3,3,3,3,3,3,3}; //{0,1,2,3,4,5,6,7};

    // vector< vector<float>> w21 = prj21->getstateij("Wij");
    // for (auto i: tempw) {
    //     for (auto j: i)
    //         std::cout<<j<<" ";
    //     std::cout<<std::endl;
    // }
    // std::cout<<std::endl;

    // vector<float> tempb = prj11->getstatej("Bj");
    // float m = calculateMean(tempb);
    // tempb = prj21->getstatej("Bj")
    // setmode("encode","alla");
    // prj11->setstate("Bj",0);
    // prj21->setstate("Bj",0);
    // resetstate();



    // std::map<int, vector<int> > assocs = getOdorwiseAssocs();
    // w21 = modify_od2lang_weights(w21,1,assocs,1.46875);

    int silentHCs;
    vector<vector<float>> cuepats;
    if (cued_net=="LTM1") {

        printf("Testing ltm2 (odor) -->ltm1 (lang) association (%d)\n",simstep);
        setmode("recall","alla");

        // //Set LangNet (LTM1) biases uniform to the average bias in the network
        // prj11->setstate("Bj",m);
        // prj21->setstate("Bj",m);

        // //Set Odor to lang weights w21 after matching 2 and 3 assoc weight distribution
        // prj21->setstate("Wij",w21);

        simstages.push_back(-2);
        simstages.push_back(simstep);


        if (distortHCs2>0) {
            // std:cout<<"FLAG!!!";
            cuepats = distort_pats(trpats2,H2,M2,"distortHCs_ltm2.bin",distortHCs2,0,1);
            // cuepats = get_noisy_pats(cuepats,0.2);
        }
        else if (cueHCs>0) {
                
            cuepats = make_custom_partialcues(partial_mode,"LTM2",H2,M2,intensity_based_units);
        }
        else
            cuepats = trpats2; //get_noisy_pats(trpats2,0.3); //trpats2; get_noisy_pats(trpats2,0.3);
 
        // // // Scale input patterns

        float k = 1;

        // for (int i = 0; i<cuepats.size();i++)
        //     for (int j = 0; j<cuepats[i].size();j++) {
        //         if (cuepats[i][j]>0.45)
        //             cuepats[i][j] *= k;
        //     }

        // prj11->prnstate("Bj");

        vector<int> v;
        float j = 0;
        for (int i = 0;i < cues.size();i++)
        {
            resetstate();
            // // Modify recuwgain for specific cues
            // v = assocs.at(cues[i]);
            // if (v.size()==2) {
            //     // prj12->setparam(WGAIN,assowgain/1.418);
            //     prj21->setparam(WGAIN,assowgain/1.2647);
            // }
            // else {
            //     // prj12->setparam(WGAIN,assowgain);
            //     prj21->setparam(WGAIN,assowgain);
            // }
            // std::cout<<"Cue "<<i+1<<" : "<<ODORS[cues[i]]<<std::endl;
            ltm1->setparam(IGAIN,0); ltm2->setparam(IGAIN,igain); ////Modulate input gain only while providing input stimulus

            if (use_intensity==1) {
            ///// If creating partial pattern based on intensity ratings
                silentHCs = get_odintensityHCs(i);
                ltm2->setinp(partialize_pattern(trpats2[cues[i]],silentHCs,H2,k,1));
            }
            else
            {
                ltm2->setinp(cuepats[cues[i]]);
            }


            //Lower recuwgain to match 3 assocs weight distribution in language net

            simulate(recallnstep);
            simstages.push_back(simstep);


            ltm1->setparam(IGAIN,0); ltm2->setparam(IGAIN,0);
            ltm1->setinp(0); ltm2->setinp(0);
            simulate(recallngap);
            simstages.push_back(simstep);

        }

    

        // for (int i = 0; i<cuepats.size();i++)
        //     for (int j = 0; j<cuepats[i].size();j++) {
        //         if (cuepats[i][j]!=0)
        //             cuepats[i][j] = 1;
        //     }
        fwritemat(recallphase_partialpats,"recallphase_partialpats.bin");
        fwritemat(cues,"cues.bin");
        fwritemat(cuepats,"cuepats.bin");

    }

    else if (cued_net=="LTM2") {

        printf("Testing ltm1 (lang) -->ltm2 (od) association (%d)\n",simstep);
        setmode("recall","alla");
        simstages.push_back(-2);
        simstages.push_back(simstep);

        // ltm1->setparam(NMEAN,nmean); ltm2->setparam(NMEAN,nmean);
        ltm1->setparam(IGAIN,igain); ltm2->setparam(IGAIN,0);
        // ltm1->setparam("namp",namp); ltm2->setparam("namp",namp);

        if (distortHCs1>0) 
            cuepats = distort_pats(trpats1,H,M,"distortHCs_ltm1.bin",distortHCs1);
        else
            cuepats = trpats1;

        for (int i = 0;i < cues.size();i++)
        {
            resetstate();
            // printf("Cueing pattern no %d (%d)\n",i,simstep);
            //ltm1->setinp(tester[i]);
            //ltm1->setinp(0);
            ltm1->setinp(cuepats[cues[i]]);
            simulate(recallnstep);
            simstages.push_back(simstep);
            ltm1->setinp(0); ltm2->setinp(0);
            simulate(recallngap);
            simstages.push_back(simstep);

        }

        fwritemat(cues,"cues.bin");
        fwritemat(cuepats,"cuepats.bin");
    }

    else  throw std::runtime_error("recall2nets: Invalid cued_net!");


}

void recall_cuebothnets() {
    // // // Cue both nets simultaneously
    vector<float> cues = {0};
    vector<vector<float>> cuepats1,cuepats2;

    printf("Testing ltm2 (odor) -->ltm1 (lang) association (%d)\n",simstep);
    setmode("recall","alla");
    simstages.push_back(-2);
    simstages.push_back(simstep);

    // ltm1->setparam(NMEAN,nmean); ltm2->setparam(NMEAN,nmean);
    ltm1->setparam(IGAIN,igain); ltm2->setparam(IGAIN,igain);
    //ltm1->setparam("taum",0.005); ltm2->setparam("taum",0.005)
        cuepats1 = trpats1;
        cuepats2 = trpats2;
    float k = 100;

    for (int i = 0; i<cuepats1.size();i++)
        for (int j = 0; j<cuepats1[i].size();j++) {
            if (cuepats1[i][j]!=0)
                cuepats1[i][j] = k;
        }

    for (int i = 0; i<cuepats2.size();i++)
        for (int j = 0; j<cuepats2[i].size();j++) {
            if (cuepats2[i][j]!=0)
                cuepats2[i][j] = k;
        }

            
    for (int i = 0;i < cues.size();i++)
    {
        resetstate();
        // ltm2->setparam(BWGAIN,0);
        ltm1->setinp(cuepats1[cues[i]]);
        ltm2->setinp(cuepats2[cues[i]]);
        simulate(recallnstep);
        simstages.push_back(simstep);
        // ltm1->setparam(BWGAIN,bwgain); ltm2->setparam(BWGAIN,bwgain);
        ltm1->setinp(1e2); ltm2->setinp(1e2);
        simulate(recallngap);
        simstages.push_back(simstep);

    }
    fwritemat(cues,"cues.bin");
    fwritemat(cuepats2,"cuepats.bin");

}

void recall_patcompletion() {
    // Test pattern completion of each networks by cueing a certain number of HCs in both

    // // Create partial patterns as cues
    vector<vector<float>> cues1; //(trpats1.size(), vector<float> (patsize1,0));
    vector<vector<float>> cues2; //(trpats2.size(), vector<float> (patsize2,0));

    if (cueHCs>0 && (distortHCs1==0 || distortHCs2==0)) {

    cues1 = make_custom_partialcues(partial_mode,"LTM1",H,M);
    cues2 = make_custom_partialcues(partial_mode,"LTM2",H2,M2);
    }
    else if (distortHCs1>0 && distortHCs2==0) {
        // std::cout<<"FLAG!!!!! 1";
        cues1 = distort_pats(trpats1,H,M,"distortHCs_ltm1.bin",distortHCs1);
        cues2 = trpats2;
    }
    else if (distortHCs2>0 && distortHCs1==0) {
        // std::cout<<"FLAG!!!!! 2";
        cues2 = distort_pats(trpats2,H2,M2,"distortHCs_ltm2.bin",distortHCs2);
        cues1 = trpats1;
    }
    else if (distortHCs2>0 && distortHCs1>0) {
        // std::cout<<"FLAG!!!!! 3";
        cues1 = distort_pats(trpats1,H,M,"distortHCs_ltm1.bin",distortHCs1);
        cues2 = distort_pats(trpats2,H2,M2,"distortHCs_ltm2.bin",distortHCs2);
    }
    else {
        // std::cout<<"FLAG!!!!! 4";
        cues1 = trpats1;
        cues2 = trpats2;
    }
    
    setmode("recall","recu");
    printf("Testing ltm1 (lang) and ltm2 (od) pattern completion, cueHCs: %d  (%d)\n",cueHCs,simstep);
    ltm1->setparam(IGAIN,igain); ltm2->setparam(IGAIN,igain);
    simstages.push_back(-2);
    simstages.push_back(simstep);
    vector<float> cue_sequence = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    int unit;
    // for (int i=0;i<trpats1.size();i++) {

    //     for (int h=0;h<cueHCs;h++) {
    //         for(int m=0;m<M;m++) {
    //             unit = h*M+m;
    //             partial_cues1[i][unit] = trpats1[i][unit];
    //         }
    //     }
    // }

    // for (int i=0;i<trpats2.size();i++) {

    //     for (int h=0;h<cueHCs;h++) {
    //         for(int m=0;m<M2;m++) {
    //             unit = h*M2+m;
    //             partial_cues2[i][unit] = trpats2[i][unit];
    //         }
    //     }
    // }


    //Works only if same number of patterns in both networks
    for (int i = 0;i < cues1.size();i++)
    {
        resetstate();
        printf("Cueing pattern no %d (%d)\n",i,simstep);
        ltm1->setinp(cues1[cue_sequence[i]]);
        ltm2->setinp(cues2[cue_sequence[i]]);
        simulate(recallnstep);
        simstages.push_back(simstep);
        ltm1->setinp(0); ltm2->setinp(0);
        simulate(recallngap);
        simstages.push_back(simstep);

    }
    fwritemat(cue_sequence,"cues.bin");
    fwritemat(cues1,"cues1.bin");
    fwritemat(cues2,"cues2.bin");


}


void cong_incong_recall() {

    vector<float> cues = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    // // // (valid only for 16 odor-16 labels case)
    // vector<float> ltm1_cues = cues;
    // vector<float> ltm1_cues = {7,6,11,5,10,3,1,6,15,2,12,2,5,4,7,8}; //Most similar in language space 
    // vector<float> ltm1_cues = {7,7,10,2,10,10,3,0,15,1,12,1,10,10,15,14}; //Most similar in odor space
    vector<float> ltm1_cues = {10,15,15,15,0,0,15,4,1,12,0,0,9,0,0,2}; //Least Similar in language space
    // vector<float> ltm1_cues = {4,8,8,8,0,15,8,15,10,0,8,15,8,8,2,5}; //Least similar in odor space
    vector<int> cong_incong_recallstages;
    vector<vector<float>> cuepats;
    if (cued_net=="LTM1") {

        printf("Testing ltm2 (odor) -->ltm1 (lang) association (%d)\n",simstep);
        setmode("recall","asso21");
        simstages.push_back(-2);
        simstages.push_back(simstep);

        // ltm1->setparam(NMEAN,nmean); ltm2->setparam(NMEAN,nmean);
        ltm1->setparam(IGAIN,igain); ltm2->setparam(IGAIN,igain);
        //ltm1->setparam("taum",0.005); ltm2->setparam("taum",0.005);

        if (distortHCs2>0) {
            // std:cout<<"FLAG!!!";
            cuepats = distort_pats(trpats2,H2,M2,"distortHCs_ltm2.bin",distortHCs2);
        }
        else if (cueHCs>0) {
            cuepats = make_custom_partialcues(partial_mode,"LTM2",H2,M2);
        }
        else
            cuepats = trpats2;

        for (int i = 0;i < cues.size();i++)
        {
            resetstate();

            //Stimulate Odor Cue
            ltm2->setparam(BWGAIN,0); ltm1->setparam(BWGAIN,bwgain);
            ltm2->setinp(cuepats[cues[i]]);
            simulate(recallnstep/2);
            simstages.push_back(simstep);
            cong_incong_recallstages.push_back(simstep); 

            //Lift all input
            ltm2->setparam(BWGAIN,bwgain); ltm1->setparam(BWGAIN,bwgain);
            ltm1->setinp(0); ltm2->setinp(0);
            simulate(3*recallngap/4);
            cong_incong_recallstages.push_back(simstep);    //Store simsteps for plotting vertical lines in main raster plot

            //Stimulate Lang Cue
            ltm2->setparam(BWGAIN,bwgain); ltm1->setparam(BWGAIN,0);
            ltm1->setinp(trpats1[ltm1_cues[i]]); ltm2->setinp(0);
            simulate(recallnstep/2);
            cong_incong_recallstages.push_back(simstep);

            //Lift All Input
            ltm1->setparam(BWGAIN,bwgain); ltm2->setparam(BWGAIN,bwgain);
            ltm1->setinp(0); ltm2->setinp(0);
            simulate(recallngap/4);
            simstages.push_back(simstep);


        }

        fwritemat(cues,"cues.bin");
        fwritemat(ltm1_cues,"ltm1_cues.bin");
        fwritemat(cuepats,"cuepats.bin");
        fwritevec(cong_incong_recallstages,"cong_incong_recallstages.txt");
    }

    else if (cued_net=="LTM2") {

        //NEED TO UPDATE LTM2 BLOCK

        printf("Testing ltm1 (lang) -->ltm2 (od) association (%d)\n",simstep);
        setmode("recall","asso21");
        simstages.push_back(-2);
        simstages.push_back(simstep);

        // ltm1->setparam(NMEAN,nmean); ltm2->setparam(NMEAN,nmean);
        ltm1->setparam(IGAIN,igain); ltm2->setparam(IGAIN,0);
        // ltm1->setparam("namp",namp); ltm2->setparam("namp",namp);

        if (distortHCs1>0) 
            cuepats = distort_pats(trpats1,H,M,"distortHCs_ltm1.bin",distortHCs1);
        else
            cuepats = trpats1;

        for (int i = 0;i < cues.size();i++)
        {
            resetstate();
            // printf("Cueing pattern no %d (%d)\n",i,simstep);
            //ltm1->setinp(tester[i]);
            //ltm1->setinp(0);
            ltm1->setparam(BWGAIN,bwgain);
            ltm1->setinp(cuepats[cues[i]]);
            simulate(recallnstep);
            simstages.push_back(simstep);
            ltm1->setparam(BWGAIN,bwgain); ltm2->setparam(BWGAIN,bwgain);
            ltm1->setinp(0); ltm2->setinp(0);
            simulate(recallngap);
            simstages.push_back(simstep);

        }

        fwritemat(cues,"cues.bin");
        fwritemat(cuepats,"cuepats.bin");

    }

    else  throw std::runtime_error("recall2nets: Invalid cued_net!");

}

void free_recall() {
        setmode("recall","alla");
        printf("Performing Free Recall (%d)\n",simstep);
        simstages.push_back(-2);
        simstages.push_back(simstep);

        for (int i = 0;i < 1;i++)
        {
            resetstate();
            // ltm2->setparam(BWGAIN,1);
            // ltm1->setparam(BWGAIN,bwgain); ltm2->setparam(BWGAIN,bwgain);
            ltm1->setinp(1e2); ltm2->setinp(1e2);
            simulate(recallngap);
            simstages.push_back(simstep);

        }

        //  // //Add varying length of cues
        // float recuwgn = 1;
        // for (int i = 0;i < 10;i++)
        // {
        //     resetstate();

        //     // ltm2->setparam(BWGAIN,1);
        //     // ltm1->setparam(BWGAIN,bwgn); ltm2->setparam(BWGAIN,bwgn);
        //    // prj11->setparam(WGAIN,recuwgn); prj22->setparam(WGAIN,recuwgn);
        //     if (i == 0) {
        //     ltm1->setinp(1e2); ltm2->setinp(1e2); }
        //     else {
        //     ltm1->setinp(trpats1[i]); ltm2->setinp(trpats2[i]);   
        //     }
        //     simstages.push_back(simstep);
        //     simulate(i*50);
        //     simstages.push_back(simstep);
        //     ltm1->setinp(1e2); ltm2->setinp(1e2);
        //     simulate(recallngap/10);

        //     recuwgn += 1;

        // }


        // // Tweak a parameter during recall 
        // float agn = 0.2;
        // for (int i = 0;i < 10;i++)
        // {
        //     resetstate();

        //     // ltm2->setparam(BWGAIN,1);
        //     ltm1->setparam("again",agn); ltm2->setparam("again",agn);
        //     // prj11->setparam("wgain",recuwgn); prj22->setparam("wgain",recuwgn);
        //     ltm1->setinp(1e2); ltm2->setinp(1e2);
        //     simulate(recallngap/10);
        //     simstages.push_back(simstep);
        //     agn += 0.5*agn;

        // }



}
void recall_extendedcue() {
    /*
        Each cue lasts throughout while in the other network, we present brief congruent & incongruent cues wrt extended cue in other network
    */



    vector<vector<float>> cuepats;
    if (cued_net=="LTM1") {
        vector<float> cues = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
        // vector<float> cues = {15,15,15};

        //alternatives are arranged in order of stored patterns, See Figsolflang.generate_alternatives() for creation
        vector<vector<float>> alternatives = { {10, 1, 7, 0},
            {15, 14, 7, 1},
            {15, 4, 11, 2},
            {15, 1, 5, 3},
            {0, 14, 12, 4},
            {0, 6, 3, 5},
            {15, 11, 7, 6},
            {4, 9, 6, 7},
            {1, 7, 15, 8},
            {12, 7, 6, 9},
            {0, 14, 12, 10},
            {0, 12, 2, 11},
            {0, 7, 5, 12},
            {0, 2, 11, 13},
            {0, 15, 7, 14},
            {1, 14, 8, 15}
        };

        vector<int> extendedcue_recallstages;
        printf("Testing ltm2 (odor) -->ltm1 (lang) association (%d)\n",simstep);
        setmode("recall","asso");
        simstages.push_back(-2);
        simstages.push_back(simstep);


        ltm2->setparam(IGAIN,igain); ltm1->setparam(IGAIN,igain);
        for (int i = 0;i < cues.size();i++)
        {
            resetstate();
            ltm1->setinp(0); ltm2->setinp(trpats2[cues[i]]);
            prj11->setparam(IWGAIN,recuwgain*10);
            // prj22->setparam(IWGAIN,recuwgain*10);
            prj12->setparam(IWGAIN,assowgain*10);
            prj21->setparam(IWGAIN,assowgain*10);
            simulate(recallnstep/5,true,false);
            for (int j = 0;j<alternatives[i].size();j++) 
            {
                // resetstate();
                //Brief alternative cue
                extendedcue_recallstages.push_back(simstep);
                ltm1->setinp(trpats1[alternatives[i][j]]); ltm2->setinp(trpats2[cues[i]]);
                if (j != 3) {
                    prj11->setparam(IWGAIN,recuwgain*10);
                    // prj22->setparam(IWGAIN,recuwgain*10);
                    prj12->setparam(IWGAIN,assowgain*10);
                    prj21->setparam(IWGAIN,assowgain*10);
                }
                else {
                    prj11->setparam(IWGAIN,recuwgain);
                    // prj22->setparam(IWGAIN,recuwgain);
                    prj12->setparam(IWGAIN,assowgain);
                    prj21->setparam(IWGAIN,assowgain);
                }
                simulate(recallngap/4,true,false);
                extendedcue_recallstages.push_back(simstep);
                //Take out alternative cue
                ltm1->setinp(0); ltm2->setinp(trpats2[cues[i]]);
                simulate(recallnstep/5,true,false);
            }


        }

        fwritemat(cues,"cues.bin");
        fwritemat(alternatives,"alternatives.bin");
        // fwritemat(cuepats,"cuepats.bin");
        fwritevec(extendedcue_recallstages,"extendedcue_recallstages.txt");

    }


    else  throw std::runtime_error("recall2nets: Invalid cued_net!");

}

void recall_sniffcues() {
    /*
        Simulate sniffing by presenting same odor cue multiple times in a given cue period
    */
    setmode("recall","alla");
    printf("Performing Multi-Cued Recall (%d)\n",simstep);
    simstages.push_back(-2);
    simstages.push_back(simstep);
    ltm1->setparam(IGAIN,0); ltm2->setparam(IGAIN,igain);
    int sniffs = 5; // Number of times to present odor cue
    vector<float> cues = {7};//{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    vector<vector<float >> cuepats = trpats2;

    // // // Scale input patterns
    float k = 100;

    for (int i = 0; i<cuepats.size();i++)
        for (int j = 0; j<cuepats[i].size();j++) {
            if (cuepats[i][j]!=0)
                cuepats[i][j] = k;
        }

    for (auto i: cuepats[0])
            std::cout<<i<<" ";

    for (int i = 0;i < cues.size();i++)
    {
        resetstate();

        for (int j = 0;j<sniffs;j++) {
            ltm2->setinp(cuepats[cues[i]]);
            simulate(recallnstep/sniffs);
            simstages.push_back(simstep);
            ltm1->setinp(1e2); ltm2->setinp(1e2);
            simulate(recallngap/sniffs);
            simstages.push_back(simstep);
        }
    }

    fwritemat(cues,"cues.bin");
    fwritemat(cuepats,"cuepats.bin");

}
/*
void recall() {

        
    
    // vector<vector<int>> cueID = getActiveUnits();
    // vector<vector<float>> cue(cueID.size(), vector<float> (patsize));

    // vector<int> active_units;
    // for (int i = 0;i<cueID.size();i++)
    // {
    //     active_units = cueID[i];
    //     std::cout<<"\nActive unit: "<<active_units[0];
    //     for (int j = 0;j<patsize;j++){ 
    //         if (j==active_units[0]*2) cue[i][j] = 1.0;
    //         else cue[i][j] = 0.0;
    //     }
    // }

    simstages.push_back(-2);
    simstages.push_back(simstep);

    // Create partial patterns as cues
    // vector<vector<float>> cue(trpats[0].size(), vector<float> (patsize));

    // for (int i=0;i<trpats[0].size();i++) {
    //     int counter = 0;
    // for (auto j: trpats[0][i]) {    //trpats[0].size()-1-i 
    //     if (counter < trpats[0][i].size()/4)
    //         cue[i][counter] = j;
    //     else
    //         cue[i][counter] = 0;
    //     counter++;
    // }
    // }

    // // // Cue partial pattern //////
    // vector<vector<float>> cues(trpats.size(), vector<float> (patsize,0));
    // int unit;
    // for (int i=0;i<trpats.size();i++) {

    //     for (int h=0;h<cueHCs;h++) {
    //         for(int m=0;m<M;m++) {
    //             unit = h*M+m;
    //             cues[i][unit] = trpats[i][unit];
    //         }
    //     }
    // }


    int distortHCs = 8;
    vector<vector<float>> cues(trpats1.size(), vector<float> (patsize,0));
    cues = distort_pats(trpats1,distortHCs);

    // vector<vector<float>> cue(trpats.size(), vector<float> (patsize));

    // for (int i = 0;i<trpats.size();i++) {
    //     int counter = 0;
    //     for(auto j:trpats[i]) {
    //         if (counter<trpats[i].size()/4)
    //             cue[i][counter]=j;
    //         else 
    //             cue[i][counter]=0;
    //         counter++;
    //     }
    // }

    ltm1->setparam(IGAIN,igain);
    setmode("recall","recu");
    setPRN(0);

    printf("Recall Phase (%d) \n",simstep);

    for(int i = 0;i<cues.size();i++) {
    resetstate();
    // ltm1->setinp(cue[cue.size()-1-i]); //Reverse
    ltm1->setinp(cues[i]);
    printf("Cueing pattern no %d (%d)\n",i,simstep);
    simulate(recallnstep,true,true);   // Dont log weights and biases becaues they are static in recall phase
    simstages.push_back(simstep);
    ltm1->setinp(0);
    simulate(recallngap,true,true);
    simstages.push_back(simstep);
    }

    fwritemat(cues,"cues.bin");

}
*/

void run(int argc,char **args) {


    Timer *alltimer = new Timer("Time elapsed");


    int sub_flag = 1;   //Set to 0 if load all subjects or 1 for single subject patterns 

    setupLTM(argc,args,sub_flag);

    if (runflag == "encode_local_only")
        trainRecu();
    else if(runflag == "encode_local_asso") {
        trainRecu();
        trainAssoc();
    }
    else if (runflag =="full") {
        trainRecu();
        trainAssoc();
        // train_all_prjs();
        // free_recall();
        recall2nets();
        //recall_patcompletion();
        // recall_sniffcues();
    }
    else if (runflag == "preload_local") {
        preloadBW("local");
        trainAssoc(); 
        // free_recall();
        recall2nets();
        // recall_extendedcue();
        //cong_incong_recall();
        //recall_patcompletion();
       
    }
    else if (runflag == "preload_localasso") {
        preloadBW("localasso");
        //free_recall();
        recall2nets();
        // recall_extendedcue();
        //cong_incong_recall();
        //recall_patcompletion();
        // recall_sniffcues();
    }

    fwritemat(trpats1,"trpats1.bin");

    fwritemat(trpats2,"trpats2.bin");

    fwritevec(simstages,"simstages.txt");


    if (prj11!=NULL) {

        printf("Logging Weights and Biases (%d)\n",simstep);

        prj11->fwritestate("Bj","Bj11.bin");

        prj11->fwritestate("Wij","Wij11.bin");


    }

    if (prj22!=NULL) {

        // printf("Logging Weights and Biases (%d)\n",simstep);

        prj22->fwritestate("Bj","Bj22.bin");

        prj22->fwritestate("Wij","Wij22.bin");

    }

    if (prj12!=NULL) {

        // printf("Logging Weights and Biases (%d)\n",simstep);

        prj12->fwritestate("Bj","Bj12.bin");

        prj12->fwritestate("Wij","Wij12.bin");

        prj12->fwritestate("Pij","Pij12.bin");
    }

    if (prj21!=NULL) {

        // printf("Logging Weights and Biases (%d)\n",simstep);

        prj21->fwritestate("Bj","Bj21.bin");

        prj21->fwritestate("Wij","Wij21.bin");

        prj21->fwritestate("Pij","Pij21.bin");

        prj21->fwritestate("Pj","Pj21.bin");

        prj21->fwritestate("Pi","Pi21.bin");

    }

        // prj11->fwritestate("Won","Won11.bin");




    Logger::closeall();

    alltimer->print();

    printf("Time simulated = %.3f sec (%d steps)\n",simstep*timestep,simstep); 

}

int main(int argc,char **args) { 

    run(argc,args);

   
} 
    

