/*

Project description

*/


#include <stdlib.h>
#include <vector>
#include <string>
#include <numeric>
#include <random>
#include <limits>
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
    bwgain = 1,pdens = 1,wdens = 1,cspeed = 1,recuwgain = 0,assowgain = 1, p0 = 1, tausd = 0, tausf = 0,assocpdens=1, epsfloat_multiplier = 1;
int nstep = 1,ngap = 1,npat1 = 1,npat2 = 1,etrnrep = 1,atrnrep = 1, recallnstep=100,recallngap=100;
int biased_item = -1, biased_context = -1, biased_assoc = -1,cueHCs = 5,distortHCs=0;
string etrn_bias_mode = "none"; // for training LTM1 and 2. stim_length, kappa, none
string pattstr = "ortho",modestr = "resetstate",wgainstr = "onlyassoc", runflag = "full", encode_order="normal",cued_net="LTM1",partial_mode="uniform";
Pat_t pattype = ORTHO;
vector<vector<int> > patstat;
vector<int> simstages;
// vector<vector<vector<float>>> trpats;

vector<vector<float>> trpats1,trpats2;

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
    parseparam->postparam("wdens",&wdens,Float);
    parseparam->postparam("cspeed",&cspeed,Float);
    parseparam->postparam("taucond",&taucond,Float);
    parseparam->postparam("taup",&taup,Float);
    parseparam->postparam("recuwgain",&recuwgain,Float);
    parseparam->postparam("assowgain",&assowgain,Float);
    parseparam->postparam("bgain",&bgain,Float);
    parseparam->postparam("bwgain",&bwgain,Float);
    parseparam->postparam("pattstr",&pattstr,String);
    parseparam->postparam("etrnrep",&etrnrep,Int);
    parseparam->postparam("nstep",&nstep,Int);
    parseparam->postparam("recallnstep",&recallnstep,Int);
    parseparam->postparam("recallngap",&recallngap,Int);
    parseparam->postparam("ngap",&ngap,Int);
    parseparam->postparam("encode_order",&encode_order,String);
    parseparam->postparam("runflag",&runflag,String);
    parseparam->postparam("epsfloat_multiplier",&epsfloat_multiplier,Float);
    parseparam->postparam("cued_net",&cued_net,String);
    parseparam->postparam("partial_mode",&partial_mode,String);
    parseparam->postparam("cueHCs",&cueHCs,Int);
    parseparam->postparam("distortHCs",&distortHCs,Int);
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
    prj11->setparam(BGAIN,bgain); prj22->setparam(BGAIN,bgain);
    prj12->setparam(BGAIN,bgain); prj21->setparam(BGAIN,bgain);

    } else if (wgainstr=="zero") {

    prj11->setparam(WGAIN,0); prj22->setparam(WGAIN,0);
    prj12->setparam(WGAIN,0); prj21->setparam(WGAIN,0);
	

    } else if (wgainstr=="recu") {

    prj11->setparam(WGAIN,recuwgain); prj22->setparam(WGAIN,recuwgain);
    prj12->setparam(WGAIN,0); prj21->setparam(WGAIN,0);

  }   else if (wgainstr=="asso") {

    prj12->setparam(WGAIN,assowgain); prj21->setparam(WGAIN,assowgain);

    } else if (wgainstr=="asso12") {

    prj11->setparam(WGAIN,0); prj22->setparam(WGAIN,0);
    prj12->setparam(WGAIN,assowgain); prj21->setparam(WGAIN,0);

    } else if (wgainstr=="asso21") {

    prj11->setparam(WGAIN,0); prj22->setparam(WGAIN,0);
    prj12->setparam(WGAIN,0); prj21->setparam(WGAIN,assowgain);

    }  else error("emmain1::setwgain","Illegal wgainstr");


}


void resetstate() {

    Pop::resetstateall(); Prj::resetstateall();
     // ltm1->resetstate(); prj11->resetstate();

}


void setmode(string modestr,string wgainstr = "zero") {

    if (modestr=="encode") {

	ltm1->setparam(BWGAIN,0); ltm2->setparam(BWGAIN,0);

	prj11->setparam(PRN,1); prj22->setparam(PRN,1); prj12->setparam(PRN,1); prj21->setparam(PRN,1);

    ltm1->setparam(ADGAIN,adgain); ltm2->setparam(ADGAIN,adgain);

    } else if (modestr=="encoderecu") {

	ltm1->setparam(BWGAIN,0); ltm2->setparam(BWGAIN,0);

    ltm1->setparam(ADGAIN,adgain); ltm2->setparam(ADGAIN,adgain);

	ltm1->setparam(AGAIN,again); ltm2->setparam(AGAIN,again);

	prj11->setparam(PRN,1); prj22->setparam(PRN,1); prj12->setparam(PRN,0); prj21->setparam(PRN,0);

    } else if (modestr=="encodeasso") {

	ltm1->setparam(BWGAIN,0); ltm2->setparam(BWGAIN,0);

	ltm1->setparam(AGAIN,again); ltm2->setparam(AGAIN,again);

    ltm1->setparam(ADGAIN,adgain); ltm2->setparam(ADGAIN,adgain);

	prj11->setparam(PRN,0); prj22->setparam(PRN,0); prj12->setparam(PRN,1); prj21->setparam(PRN,1);

    } else if (modestr=="recall") {

	ltm1->setparam(BWGAIN,bwgain); ltm2->setparam(BWGAIN,bwgain);

	ltm1->setparam(ADGAIN,adgain); ltm2->setparam(ADGAIN,adgain);

	ltm1->setparam(AGAIN,again); ltm2->setparam(AGAIN,again);

    // ltm1->setparam("thres",0.5);    ltm2->setparam("thres",0.5);

	prj11->setparam(PRN,0); prj22->setparam(PRN,0); prj12->setparam(PRN,0); prj21->setparam(PRN,0);

    } else if (modestr=="resetstate") resetstate();

    else error("olflangmain1::setmode","Illegal modestr");

    setwgain(wgainstr);

}



/***** Use for custom initiation P-traces and Wij/Bj */


void initPWB(string prj, vector<vector<float> > w) {

  float P = 1;
  vector<float> Pi(H*M),Pj(H*M);
  vector<vector<float> > Pij(H*M,vector<float>(H*M));

  for (int i=0; i<H*M; i++) Pi[i] = gnextfloat();

  for (int j=0; j<H*M; j++) Pj[j] = gnextfloat();

  for (int i=0; i<H*M; i++)

    for (int j=0; j<H*M; j++) 
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

void initPWB(string prj, vector<vector<float> > w, vector<float> b ,int HC,int MC) {

  float P = 1;
  vector<float> Pi(HC*MC),Pj(HC*MC);
  vector<vector<float> > Pij(HC*MC,vector<float>(HC*MC));

  for (int i=0; i<HC*MC; i++) Pi[i] = gnextfloat(); 

  for (int j=0; j<HC*MC; j++) Pj[j] = exp(b[j]);//gnextfloat();

  for (int i=0; i<HC*MC; i++)

    for (int j=0; j<HC*MC; j++) {

        Pij[i][j] = Pi[i]*Pj[j]/P * exp(w[i][j]/recuwgain);

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

    ltm2->logstate("inp","inp2.log");
    ltm2->logstate("dsup","dsup2.log");
    ltm2->logstate("act","act2.log");
    ltm2->logstate("bwsup","bwsup2.log");
    // ltm1->logstate("ada","ada1.log");
    ltm1->logstate("expdsup","expdsup1.log");
    ltm2->logstate("expdsup","expdsup2.log");


    if (prj11!=NULL) prj11->logstate("Bj","Bj11.log");
    if (prj11!=NULL) prj11->logstate("Wij","Wij11.log");

    if (prj22!=NULL) prj22->logstate("Bj","Bj22.log");
    if (prj22!=NULL) prj22->logstate("Wij","Wij22.log");

    if (prj12!=NULL) prj12->logstate("Bj","Bj12.log");
    if (prj12!=NULL) prj12->logstate("Wij","Wij12.log");

    if (prj21!=NULL) prj21->logstate("Bj","Bj21.log");
    if (prj21!=NULL) prj21->logstate("Wij","Wij21.log");

    // if (prj11!=NULL) prj11->logstate("Pij","Pij11.log");
    // if (prj11!=NULL) prj11->logstate("Pi","Pi11.log");
    // if (prj11!=NULL) prj11->logstate("Pj","Pj11.log");   
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

    }

    npat1 = imax1 + 1;

    npat2 = imax2 + 1;


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



vector<vector<float> > readbinmat(string filename, int n)
{
  //read matrix from .bin file
  int m = n*n;
  vector<float> state; //1d vector
  state.resize(m);
  vector<vector<float> > state2d; //2d vector of size NxN
  state2d.resize(n);
  for(int i = 0;i<n;i++)
  {
    state2d[i].resize(n);
  }
  ifstream inf;
  inf.open(filename.c_str(),ios::binary);
  if(!inf)
    error("emmain::readbinmat","Could not open file: " + filename);
  inf.read(reinterpret_cast<char *>(&state[0]), m*sizeof(state[0]));
  float f;
  for(int i = 0; i < state.size(); i++)
  {
      int row = i/n;
      int col = i%n;
      state2d[row][col] = state[i];
  }
  inf.close();
  return state2d;
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

vector<vector<float> > readJSONpats(string filename, string net_type) {
    
    // std::ifstream file(filename);
    // std::string unparsed;
    // std::getline(file, unparsed);
    
   
    // int nrow = 16, ncol = H*M, counter=0;  //100;
    
    // vector<vector<float> > pats(nrow,vector<float>(ncol,0));
    
    // int p = 0, Hyp = 0, parse = 0, m = 0;
    // for(char& c : unparsed) {
    //     if (c==']') {
    //         p += 1;
    //         Hyp = 0;
    //         parse = 0;
    //     } else if (c=='[') {
    //         parse = 1;
    //     } else if (c==',' or c==' ') { // do nothing
    //     } else {

    //         std::cout << Hyp << "*10+" << m << "=" << Hyp*10+m << " "<<endl;
    //         m = c - '0';
    //         pats[p][Hyp*M+m] = 1;
    //         Hyp += 1;
    //     }
    //     counter++;
    // }    
    
    // odors = nrow;
    // patsize = ncol;
    // return pats;

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
        // std::cout<<"Pattern no: "<<p<<endl;
        while(std::getline(s_stream,substr,','))
        {

           // m = stoi(substr); //Not working, why?

           // 
           parse = 1;
           bool whiteSpacesOnly = substr.find_first_not_of (' ') == substr.npos;    //Gives 1 if substr is a blank space that I get at the end of string for some reason
           //std::cout<<whiteSpacesOnly;
           if(whiteSpacesOnly == 0) {
             m = stoi(substr);
             // if (net_type == "lang")
             //    std::cout<<"m: "<<m<<"\t";
             // std::cout << Hyp << "*10+" << m << "=" << Hyp*10+m << " "<<net_type<<endl;
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

    return pats;


}

vector<vector<float>> distort_pats(vector<vector<float>> pats,int HC,int MC,int distort = 1) {

    vector<vector<float>> rp(pats.size(), vector<float>(HC*MC,0)); 

    vector<int> HCs(HC);
    int distortHC_id;
    for (int i = 0;i<HC;i++)
    {
        HCs.push_back(i);
    }
    for (int i = 0;i<pats.size();i++)
    {
        auto rng = std::default_random_engine {};
        std::shuffle(std::begin(HCs),std::end(HCs),rng);

        rp[i] = pats[i];
        for(int id = 0;id<distortHCs;id++) {
            distortHC_id = HCs[id];
            rng = std::default_random_engine {};
            std::shuffle(std::begin(rp[i])+(distortHC_id*MC),std::begin(rp[i])+(distortHC_id*MC+MC),rng);
            // std::cout<<"Distorting pattern "<<i<<" HC "<<distortHC_id<<std::endl;
        }
    }

    return rp;
}

void setupLTM(int argc,char **args,int single_sub_flag = 1) {

    ginitialize();


    string paramfile = "/home/rohan/Documents/BCPNNSimv2/works/apps/olflang/olflangmain1.par";
    // string paramfile = "/home/rohan/Documents/Olfaction/Pattern_Generation/SNACKPatterns";

    // if (argc>1) paramfile = args[1];

    parseparams(paramfile);

    if (argc>1) cueHCs = atoi(args[1]);   //If additional params given in terminal then it refers to cueHC param.


    gsetseed(seed);


    // getpatstat("/home/rohan/Documents/BCPNNSimv2/works/apps/olflang/patstat_top3snackdescs.txt");
    // getpatstat("/home/rohan/Documents/BCPNNSimv2/works/apps/olflang/patstat_n3.txt");
    getpatstat("/home/rohan/Documents/BCPNNSimv2/works/apps/olflang/patstat_16od_16descs.txt");
    

    // string binfname = "/home/rohan/Documents/BCPNNSimv2/works/apps/olflang/InpPats/artif_binpats_nonmodular_tester2(sparse)_npats5.txt";
    // string binfname = "/home/rohan/Documents/Olfaction/Pattern_Generation/SNACKPatterns/npats10_patsize303_random(modular)"; 
    // string binfname = "/home/rohan/Documents/Olfaction/Pattern_Generation/SNACKPatterns/meanpats16.txt";
    // trpats = getpats(binfname,single_sub_flag);


    //Language Network
    trpats1 = readJSONpats( "/home/rohan/Documents/Olfaction/Pattern_Generation/GMCPatterns/w2v16_D0_npat16_emax06_H10_M10.json","lang");
    // trpats1 = readJSONpats( "/home/rohan/Documents/Olfaction/Pattern_Generation/DistortedPats/npats48_distorted2_high.json","lang");
    // trpats1 = readJSONpats( "/home/rohan/Documents/Olfaction/Pattern_Generation/GMCPatterns/semanticneighbours_D0_npat48_emax05_H10_M10.json","lang");     
    // trpats1 = readJSONpats( "/home/rohan/Documents/Olfaction/Pattern_Generation/GMCPatterns/top3snackdescriptors_D0_npat35_emax06_H20_M20.json","lang"); 

    //Odor Network
    trpats2 = readJSONpats( "/home/rohan/Documents/Olfaction/Pattern_Generation/RobPats/patterns_v3(snack_sorted).json","od");

    
    trpats1 = mkpats(npat1,H,M,ORTHO),trpats2 = mkpats(npat2,H2,M2,ORTHO);

    // getpatstat("/home/rohan/Documents/BCPNNSimv2/works/apps/olflang/patstat_n3.txt");

    std::cout<<"H1: "<<H<<" M1: "<<M<<" H2: "<<H2<<" M2: "<<M2<<" npat1: "<<npat1<<" npat2: "<<npat2<<" patstat.size "<<patstat.size()<<endl;

    // printf("SUBJECTS, ODORS, PATSIZE: %d %d %d \n",subjects,odors,patsize);
    if (epsfloat_multiplier==0)
        error("olflangmain1::setupLTM","cannot epsfloat_multiplier to 0");

    else {
        epsfloat *= epsfloat_multiplier;
        if (epsfloat_multiplier!=1)     std::cout<<"\nEPSFLOAT: "<<epsfloat<<endl;

    }

    // ltm1 = new PopH(patsize/2,2,BCP,FULL);
    // ltm1 = new PopH(1,patsize,BCP,CAP);
    // ltm1 = new BCPPop(patsize,BCP,CAP);

    ltm1 = new PopH(H,M,BCP,FULL);
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


    ltm2 = new PopH(H2,M2,BCP,FULL);
    ltm2->setgeom(HEX2D);
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
    prj12->setparam("cspeed",cspeed);
    prj12->setparam("bgain",bgain);


    prj21 = new PrjH(ltm2,ltm1,assocpdens,INCR);

    prj21->setparam("wdens",wdens);
    prj21->setparam("taup",taup);
    prj21->setparam("taucond",taucond);
    prj21->setparam("cspeed",cspeed);
    prj21->setparam("bgain",bgain);


    // prj11->configwon(0);
    // simstages.push_back(odors);

    setuplogging();


}
void preloadBW() {
    // To preload stored weights and biases from .bin files instead of running encoding phase
    vector<vector<float> > Wij11, Wij22;
    vector<float> Bj11, Bj22 ;

    Wij11 = readbinmat("Wij11pre.bin",H*M);
    Wij22 = readbinmat("Wij22pre.bin",H2*M2);
    Bj11  = readbinvec("Bj11pre.bin",H*M);
    Bj22  = readbinvec("Bj22pre.bin",H2*M2);

    setmode("encoderecu","recu");

    printf("Loading weights for ltm1 and ltm2 (%d)\n",simstep);

    // prj11->prnstate("Wij");
    initPWB("prj11",Wij11,Bj11,H,M);
    initPWB("prj22",Wij22,Bj22,H2,M2);


    resetstate();

}
void trainLTMs(int init_run = 0) {



    // ltm1->setparam(ADGAIN,adgain); ltm2->setparam(ADGAIN,adgain);

    setmode("encoderecu","recu");


    ltm1->setparam(IGAIN,igain); ltm2->setparam(IGAIN,igain);
    // ltm1->setparam(NMEAN,nmean); ltm2->setparam(NMEAN,nmean);
    printf("Training ltm1 (lang) & ltm2 (odor) (%d)\n",simstep);

    if (init_run==1) {
    // initial run to stabilize weights and biasese?
    ltm1->setinp(0); ltm2->setinp(0);
    resetstate();
    simulate(nstep*2);
    }

    // vector <int> pattern_order1(npat1);
    // vector <int> pattern_order2(npat2);


    // for(int i=0; i<npat1; i++)
    //     pattern_order1[i]=i;

    // for(int i=0; i<npat2; i++)
    //     pattern_order2[i]=i;


    // if(encode_order == "random") {
    //     std::random_device rd;
    //     std::mt19937 g(rd());
    //     std::shuffle(std::begin(pattern_order1),std::end(pattern_order1),g);
    //     std::shuffle(std::begin(pattern_order2),std::end(pattern_order2),g);
    // }

    // fwritevec(pattern_order1,"ltm1_patorder.txt");
    // fwritevec(pattern_order2,"ltm2_patorder.txt");



    for(int rep = 0; rep<etrnrep; rep++) {
    for(int sub = 0; sub<1;sub++)
    {
      for(int p = 0; p<npat1; p++)
      {

         // printf("p,q: %d , %d \n",p,q);
        // printf("Encoding pattern no %d (%d)\n",od,simstep);
        resetstate();
        // setPRN(1);
        if (p >= npat2) {
            ltm2->setinp(0);
            prj22->setparam(PRN,0);
        }
        else {
            prj22->setparam(PRN,1);
            ltm2->setinp(trpats2[patstat[p][1]]);
        }
        prj11->setparam(PRN,1); 
        ltm1->setinp(trpats1[patstat[p][0]]);

        simulate(nstep,true);
        ltm1->setinp(0); ltm2->setinp(0);
        simstages.push_back(simstep);
        setPRN(0);
        simulate(ngap,true);
        simstages.push_back(simstep);
      }
    }
    }

    setPRN(0);
    // simstages.push_back(simstep);
    simulate(ngap*3,true,true);


}

void trainAssoc() {



    // vector <int> pattern_order1(npat1);
    // vector <int> pattern_order2(npat2);


    // for(int i=0; i<npat1; i++)
    //     pattern_order1[i]=i;

    // for(int i=0; i<npat2; i++)
    //     pattern_order2[i]=i;


    // if(encode_order == "random") {
    //     std::random_device rd;
    //     std::mt19937 g(rd());
    //     std::shuffle(std::begin(pattern_order1),std::end(pattern_order1),g);
    //     std::shuffle(std::begin(pattern_order2),std::end(pattern_order2),g);
    // }

    // fwritevec(pattern_order1,"assoc_patorder1.txt");
    // fwritevec(pattern_order2,"assoc_patorder2.txt");



    printf("Training ltm1<-->ltm2 association (%d)\n",simstep);
    setmode("encodeasso","asso");
    simstages.push_back(-1);
    simstages.push_back(simstep);
    // ltm1->setparam(NMEAN,nmean); ltm2->setparam(NMEAN,nmean);

    for(int rep = 0; rep<etrnrep; rep++) {
    for(int sub = 0; sub<1;sub++)
    {
      for(int p = 0; p<patstat.size();p++)
      {
        resetstate();
        prj21->setparam(PRN,1); prj12->setparam(PRN,1);
        ltm1->setinp(trpats1[patstat[p][0]]);
        ltm2->setinp(trpats2[patstat[p][1]]);
        simulate(nstep,true);
        simstages.push_back(simstep);
        ltm1->setinp(0); ltm2->setinp(0);
        setPRN(0);
        simulate(ngap,true);
        simstages.push_back(simstep);
     }
    }
    }

    setPRN(1);
    // simstages.push_back(simstep);
    simulate(ngap*3,true,true);

}

void recall2nets(int distort=0) {

    vector<float> cues = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    setPRN(0);
    vector<vector<float>> cuepats;
    if (cued_net=="LTM1") {

        printf("Testing ltm2 (odor) -->ltm1 (lang) association (%d)\n",simstep);
        setmode("recall","alla");
        simstages.push_back(-2);
        simstages.push_back(simstep);

        ltm1->setparam(NMEAN,nmean); ltm2->setparam(NMEAN,nmean);
        ltm1->setparam(IGAIN,0); ltm2->setparam(IGAIN,igain);
        // ltm1->setparam("namp",namp); ltm2->setparam("namp",namp);

        if (distort>0) {
            std:cout<<"FLAG!!!";
            cuepats = distort_pats(trpats2,H2,M2,distort);
        }
        else
            cuepats = trpats2;
        

        for (int i = 0;i < cues.size();i++)
        {
            resetstate();
            printf("Cueing pattern no %d (%d)\n",i,simstep);
            //ltm1->setinp(tester[i]);
            //ltm1->setinp(0);
            // if (i==2) {
            //     prj21->prnstate("Wij");
            // }
            ltm2->setinp(cuepats[cues[i]]);
            simulate(recallnstep,true,true);
            simstages.push_back(simstep);
            ltm1->setinp(0); ltm2->setinp(0);
            simulate(recallngap,true,true);
            simstages.push_back(simstep);

        }

        fwritemat(cues,"cues.bin");
        fwritemat(cuepats,"cuepats.bin");

    }

    else if (cued_net=="LTM2") {

        printf("Testing ltm1 (lang) -->ltm2 (od) association (%d)\n",simstep);
        setmode("recall","alla");
        simstages.push_back(-2);
        simstages.push_back(simstep);

        ltm1->setparam(NMEAN,nmean); ltm2->setparam(NMEAN,nmean);
        ltm1->setparam(IGAIN,igain); ltm2->setparam(IGAIN,0);
        // ltm1->setparam("namp",namp); ltm2->setparam("namp",namp);

        if (distortHCs>0) 
            cuepats = distort_pats(trpats1,H,M,distort);
        else
            cuepats = trpats1;

        for (int i = 0;i < cues.size();i++)
        {
            resetstate();
            printf("Cueing pattern no %d (%d)\n",i,simstep);
            //ltm1->setinp(tester[i]);
            //ltm1->setinp(0);
            ltm1->setinp(trpats1[cues[i]]);
            simulate(recallnstep,true,true);
            simstages.push_back(simstep);
            ltm1->setinp(0); ltm2->setinp(0);
            simulate(recallngap,true,true);
            simstages.push_back(simstep);

        }

        fwritemat(cues,"cues.bin");
        fwritemat(cuepats,"cuepats.bin");
    }

    else  throw std::runtime_error("recall2nets: Invalid cued_net!");


    

    // fwritemat(cues,"cues.bin");
    

}

vector<vector<float>> make_custom_partialcues(string partial_mode,string net,int HC,int MC) {
///// Cue only some hypercolumns of each pattern using list HCs 

    vector<vector<float>> trpats;
    int patsize;
    if (net=="LTM1") {
        patsize = patsize1;
        trpats = trpats1;    
    }
    else if (net=="LTM2") {
        patsize = patsize2;
        trpats = trpats2;
    }

    vector<vector<float>> partial_cues(trpats.size(), vector<float> (patsize,0));

    vector<vector<int>> active_units(trpats.size(),vector<int> (HC,0));


    if (partial_mode == "uniform") {

        vector<float> cues = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
        int unit;
        for (int i=0;i<trpats1.size();i++) {

            for (int h=0;h<cueHCs;h++) {
                for(int m=0;m<M;m++) {
                    unit = h*M+m;
                    partial_cues[i][unit] = trpats[i][unit];
                }
            }
        }
    }


    else {
        ////// Cue HCs which share most overlap or leasdt overlap over patterns
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
                cueUnits = std::vector<int>(pat.begin(), pat.begin()+cueHCs);

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

void recall_patcompletion() {
    // Test pattern completion of each networks by cueing a certain number of HCs in both

    // // Create partial patterns as cues
    vector<vector<float>> partial_cues1; //(trpats1.size(), vector<float> (patsize1,0));
    vector<vector<float>> partial_cues2; //(trpats2.size(), vector<float> (patsize2,0));

    partial_cues1 = make_custom_partialcues(partial_mode,"LTM1",H,M);
    partial_cues2 = make_custom_partialcues(partial_mode,"LTM2",H2,M2);
    setmode("recall","recu");
    printf("Testing ltm1 (lang) and ltm2 (od) pattern completion, cueHCs: %d  (%d)\n",cueHCs,simstep);
    simstages.push_back(-2);
    simstages.push_back(simstep);
    vector<float> cues = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
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
    for (int i = 0;i < partial_cues1.size();i++)
    {
        resetstate();
        printf("Cueing pattern no %d (%d)\n",i,simstep);
        ltm1->setinp(partial_cues1[cues[i]]);
        ltm2->setinp(partial_cues2[cues[i]]);
        simulate(recallnstep,true,true);
        simstages.push_back(simstep);
        ltm1->setinp(0); ltm2->setinp(0);
        simulate(recallngap,true,true);
        simstages.push_back(simstep);

    }
    fwritemat(cues,"cue_order.bin");
    fwritemat(partial_cues1,"cues1.bin");
    fwritemat(partial_cues2,"cues2.bin");


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

    std::cout<<"LTM1 H,M,Nh,N: "<<ltm1->getinfo("H")<<' '<<ltm1->getinfo("M")<<' '<<ltm1->getinfo("Nh")<<' '<<ltm1->getinfo("N")<<std::endl;

    if (runflag == "encode_only")
        trainLTMs();
    else if (runflag =="full") {
        trainLTMs();
        trainAssoc();
        if(cueHCs==0) recall2nets(distortHCs);
        else recall_patcompletion();
    }
    else if (runflag == "preload") {
        preloadBW();
        trainAssoc(); 
        if(cueHCs==0) recall2nets(distortHCs);
        else recall_patcompletion();
       
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

    }

    if (prj21!=NULL) {

        // printf("Logging Weights and Biases (%d)\n",simstep);

        prj21->fwritestate("Bj","Bj21.bin");

        prj21->fwritestate("Wij","Wij21.bin");

    }

        // prj11->fwritestate("Won","Won11.bin");




    Logger::closeall();

    alltimer->print();

    printf("Time simulated = %.3f sec (%d steps)\n",simstep*timestep,simstep); 

}

int main(int argc,char **args) { 

    run(argc,args);

    // getpatstat("/home/rohan/Documents/BCPNNSimv2/works/apps/em/patstat.txt");
    // vector<vector<float> > trpats1 = mkpats(npat1,H,M,pattype),trpats2 = mkpats(npat2,H,M,pattype);
    // vector<int> simstages;
    // vector<int> cuestages;


    // int cueHC = 274;

    // vector<float> cue(patsize);

    // for(int i=0;i<patsize/2;i++)
    // {
    //   if (i == cueHC*2)
    //   {
    //     cue[i] = 1;
    //     continue;
    //   }
    //   else if(i == cueHC*2+1)
    //   {
    //     cue[i] = 0;
    //     continue;
    //   }
    //   else
    //   {
    //     if (i%2==0)
    //       cue[i]=0;
    //     else
    //       cue[i]=1;
    //   }
    // }


    // printf("Cueing HC %d (Terpentine):  (%d)\n",cueHC,simstep);
    // // ltm1->setparam(IGAIN,igain*10);
    // setmode("recall","recu");

    // resetstate();
    // ltm1->setinp(cue);
    // simulate(recallnstep);
    // ltm1->setinp(0);
    // simulate(ngap);

   
} 
    

