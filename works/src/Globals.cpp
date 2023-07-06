#include <string>
#include <limits>
#include <random>
#include <algorithm> // transform

#include <sys/types.h>
#include <sys/stat.h>

#include "Globals.h"
#include "Pop.h"
#include "Prj.h"
#include "Logger.h"

using namespace std;
using namespace Globals;

string Globals::BCPNNSim_version = "BCPNNSim Version 0.9.5";
bool Globals::initialized = false,Globals::setupdone = false;

int Globals::simstep = 0;
float Globals::timestep = 0.001;
timespec Globals::start;

float Globals::maxfloat = +std::numeric_limits<float>::infinity();
float Globals::minfloat = -std::numeric_limits<float>::infinity();
float Globals::epsfloat = numeric_limits<float>::epsilon();

long Globals::gseed = random_device{}();

uniform_real_distribution<float> Globals::uniformfloatdistr = uniform_real_distribution<float> (0.0,1.0);

uniform_int_distribution<int> Globals::uniformintdistr = uniform_int_distribution<int>();

std::mt19937_64 Globals::generator;

double Globals::updbwtime = 0,Globals::proptime = 0,Globals::hcupdtime = 0;

vector<vector<float> > Globals::trdata,Globals::tedata,Globals::trlbl,Globals::telbl;

void Globals::ginitialize(bool printversion) {

    initialized = true;

    startnanosectimer();

    if (printversion) printf("Starting %s ...\n",BCPNNSim_version.c_str());

}


void Globals::error(std::string errloc,std::string errstr) {

    fprintf(stderr,"ERROR in %s: %s\n",errloc.c_str(),errstr.c_str());

    exit(EXIT_FAILURE);
}


void Globals::warning(std::string warnloc,std::string warnstr) {

    fprintf(stderr,"WARNING in %s: %s\n",warnloc.c_str(),warnstr.c_str());

}


void Globals::startnanosectimer() {

    // start nanosectimer
    getnanosec();
    ios_base::sync_with_stdio(false);

}


long Globals::getnanosec() {

    if (not initialized) error("Globals::getnanosec","Not initialized");

    // Read nanosectimer
    clock_gettime(CLOCK_REALTIME,&start);

    return start.tv_nsec;

}


long Globals::gsetseed(long newgseed) {

    if (newgseed==0) newgseed = random_device{}();

    gseed = newgseed%10000000;

    if (gseed<0) gseed = -gseed;

    generator.seed(gseed);

    return gseed;

}


long Globals::ggetseed() { return gseed; }


float Globals::gnextfloat() { return uniformfloatdistr(generator); }


int Globals::gnextint() { return uniformintdistr(generator); }


float Globals::distance(vector<float> xyz1,vector<float> xyz2) {

    return sqrt(pow(xyz2[0]-xyz1[0],2) + pow(xyz2[1]-xyz1[1],2) + pow(xyz2[2]-xyz1[2],2));

}


float Globals::calctaudt1(float tau) {

    float taudt = (1 - exp(-timestep/tau));  // exp(-timestep/tau)*timestep/tau;


    if (1<taudt) error("Globals::calctaudt1","Illegal taudt>1: " + to_string(taudt));

    return taudt;

}

float Globals::calctaudt(float tau) {

    if (tau<timestep) error("Globals::calctaudt","Illegal tau<timestep: " + to_string(tau));

    float taudt = timestep/tau;

    return taudt;

}


vector<vector<float> > Globals::mkpats(int Z,int H,int M,Pat_t pattype,float p1,float p0,int verbosity) {

    vector<float> pat;
    vector<vector<float> > pats;

    if (p0<0) p0 = (1-p1)/(M-1);

    for (int z=0; z<Z; z++) {

	pat = vector<float>(H*M,0);

	for (int h=0; h<H; h++) {

	    switch (pattype) {

	    case ORTHO:

		for (int m=h*M; m<(h+1)*M; m++) pat[m] = p0;

		pat[h*M + z%M] = p1;

		break;

	    case RND01:

		for (int m=h*M; m<(h+1)*M; m++) pat[m] = p0;

		pat[h*M + gnextint()%M] = p1;

		break;

	    case NOHRND01:

		for (int n=0; n<H*M; n++) pat[n] = p0;

		for (int h=0,i; h<H; h++) {

		    i = gnextint()%(H*M);

		    while (pat[i]==p1) i = gnextint()%(H*M);

		    pat[i] = p1;

		}

		break;

	    case RND:

		for (int m=h*M; m<(h+1)*M; m++) pat[m] = gnextfloat();

		break;

	    default: error("Globals::mkpat","No such pattype");

	    }

	}

	pats.push_back(pat);

    }

    if (verbosity>2) {

	for (size_t r=0; r<pats.size(); r++) prnvec(pats[r]);

	printf("\n");

    }

    return pats;

}


vector<vector<float> > Globals::readpats(int plen,string filename) {

    FILE *infp = fopen(filename.c_str(),"rb");

    vector<vector<float> > pats;

    if (infp==NULL) {

	error("Globals::readpats","Could not open file: " + filename);

    } else {

	fseek(infp,0,SEEK_END);   // non-portable

	long nbyte = ftell(infp);

	fseek(infp,0,SEEK_SET);

	if (nbyte%sizeof(float)!=0) error("Globals::readpats","Byte format error");

	int nflt = nbyte/sizeof(float);

	float *patdata = (float *)calloc(nflt,sizeof(float));

	int nread = fread(patdata,sizeof(float),nflt,infp);

	if (nread!=nflt) error("Globals::readpats","Read float error: " + to_string(nread) + " -- " + to_string(nflt));

	fclose(infp);

	/// printf ("Size of %s: %ld bytes nflt = %d nread = %d\n",filename.c_str(),nbyte,nflt,nread);

	if (nflt%plen!=0)

		error("Globals::readpats","Pat float format error: " + to_string(nflt) + " -- " + to_string(plen));

	int npat = nflt/plen;

	pats = vector<vector<float> >(npat,vector<float>(plen,0));

	for (int p=0,i=0; p<npat; p++)

		for (int c=0; c<plen; c++)

		    pats[p][c] = patdata[i++];


    }

    return pats;

}


vector<vector<float> > Globals::readpats(int npat,int plen,string filename) {

    vector<vector<float> > pats(npat,vector<float>(plen,0));
    size_t nflt;

    FILE *infp = fopen(filename.c_str(),"r");

    if (infp==NULL) error("Globals::readpats","File not found: " + filename);

    for (int p=0; p<npat; p++) nflt = fread(pats[p].data(),sizeof(float),plen,infp);

    return pats;

}


vector<vector<float> > Globals::readtxtpats(string filename) {

    FILE *infp = fopen(filename.c_str(),"r");

    if (infp==NULL) {

	warning("Globals::readtxtpats","File not found: " + filename);

	return vector<vector<float> > (0);

    }

    int nrow = 0,nit,nflt = 0; float flt;

    while (not feof(infp)) { nit = fscanf(infp,"%f",&flt); if (nit==1) nflt++; }

    rewind(infp);

    while (not feof(infp)) { nit = fscanf(infp, "%*[^\n]%*c"); if (nit==0) nrow++; }

    if (nflt%nrow!=0) error("Globals::readtxtpats","Not even lines");

    int ncol = nflt/nrow;

    /// printf("nflt = %d nrow = %d patlen = %d\n",nflt,nrow,ncol);

    vector<vector<float> > pats(nrow,vector<float>(ncol,0));

    nflt = 0;

    rewind(infp);

    for (int r=0; r<nrow; r++)

	for (int c=0; c<ncol; c++)

	    nflt += fscanf(infp,"%f ",&pats[r][c]);

    fclose(infp);

    return pats;

}


vector<vector<int> > Globals::readtxtipats(string filename) {

    FILE *infp = fopen(filename.c_str(),"r");

    if (infp==NULL) {

	warning("Globals::readtxtipats","File not found: " + filename);

	return vector<vector<int> > (0);

    }

    int nrow = 0,nit,nval = 0; int val;

    while (not feof(infp)) { nit = fscanf(infp,"%d",&val); if (nit==1) nval++; }

    rewind(infp);

    while (not feof(infp)) { nit = fscanf(infp, "%*[^\n]%*c"); if (nit==0) nrow++; }

    if (nval%nrow!=0) error("Globals::readtxtpats","Not even lines");

    int ncol = nval/nrow;

    /// printf("nval = %d nrow = %d patlen = %d\n",nval,nrow,ncol);

    vector<vector<int> > pats = vector<vector<int> >(nrow,vector<int>(ncol,0));

    nval = 0;

    rewind(infp);

    for (int r=0; r<nrow; r++)

	for (int c=0; c<ncol; c++)

	    nval += fscanf(infp,"%d ",&pats[r][c]);

    fclose(infp);

    return pats;

}


void Globals::cpyvectovec(vector<float> frvec,vector<float> &tovec,int ioffs) {

    if (ioffs<0 or ioffs+frvec.size()>tovec.size())
	error("Globals::cpyvectovec","Non-matching dimensions: " +
	      to_string(ioffs+frvec.size()) + " -- " + to_string(tovec.size()));

    for (size_t i=0; i<frvec.size(); i++) tovec[ioffs + i] = frvec[i];

}


void Globals::cpyrectomat(std::vector<std::vector<float> > &mat,std::vector<std::vector<float> > rec,int r0,int c0) {

    int mnrow = mat.size(),mncol = mat[0].size(),tnrow = rec.size(),tncol = rec[0].size();

    if (r0<0 or mnrow<r0 + tnrow)
	error("Globals::cpyrectomat","Non-matching col dimensions: " + to_string(mnrow) + " -- " +
	      to_string(r0 + tnrow));

    if (c0<0 or mncol<c0 + tncol)
	error("Globals::cpyrectomat","Non-matching col dimensions: " + to_string(mncol) + " -- " +
	      to_string(c0 + tncol));

    for (int r=0; r<tnrow; r++)

	for (int c=0; c<tncol; c++)

	    mat[r0 + r][c0 + c] = rec[r][c];

}


void Globals::cpymattorec(std::vector<std::vector<float> > mat,std::vector<std::vector<float> > &rec,int r0,int c0) {

    int mnrow = mat.size(),mncol = mat[0].size(),tnrow = rec.size(),tncol = rec[0].size();

    if (r0<0 or mnrow<r0 + tnrow) error("Globals::cpymattorec","Non-matching row dimensions");
    if (c0<0 or mncol<c0 + tncol) error("Globals::cpymattorec","Non-matching col dimensions");

    for (int r=0; r<tnrow; r++)

	for (int c=0; c<tncol; c++)

	    rec[r][c] = mat[r0 + r][c0 + c];

}


string Globals::actfn2str(Actfn_t actfn_t) {

    switch (actfn_t) {

    case NOACTFN: return "NOACTFN"; break;

    case BCP: return "BCP"; break;

    case LIN: return "LIN"; break;

    case LOG: return "LOG"; break;

    case EXP: return "EXP"; break;

    case SIG: return "SIG"; break;

    case SPKBCP: return "SPKBCP"; break;

    case SPKLIN: return "SPKLIN"; break;

    case SPKLOG: return "SPKLOG"; break;

    case SPKEXP: return "SPKEXP"; break;

    case SPKSIG: return "SPKSIG"; break;

    case AREG: return "AREG"; break;

    case ALIF: return "ALIF"; break;

    case AdEx: return "AdEx"; break;

    case AdExS: return "AdExS"; break;

    default: error("Globals::actfn2str","No such 'actfn_t'");

    }

    return "";

}


string Globals::normfn2str(Normfn_t normfn_t) {

    switch (normfn_t) {

    case NONORMFN: return "NONORMFN"; break;

    case CAP: return "CAP"; break;

    case HALF: return "HALF"; break;

    case FULL: return "FULL"; break;

    default: error("Globals::normfn2str","No such 'normfn_t'");

    }

    return "";

}


string Globals::popparam2str(Popparam_t popparam_t) {

    switch (popparam_t) {

    case TAUM: printf("taum\n"); break;
    case IGAIN: printf("igain\n"); break;
    case IBIAS: printf("ibias\n"); break;
    case AGAIN: printf("again\n"); break;
    case BWGAIN: printf("bwgain\n"); break;
    case NAMP: printf("namp\n"); break;
    case NMEAN: printf("nmean\n"); break;
    case ADGAIN: printf("adgain\n"); break;
    case TAUA: printf("taua\n"); break;
    case MAXFQ: printf("maxfq\n"); break;
    case THRES: printf("thres\n"); break;
    case SPKPER: printf("spkper\n"); break;
    case SPKPHA: printf("spkpha\n"); break;
    case ESYN: printf("Esyn\n"); break;
    case CM: printf("Cm\n"); break;
    case GL: printf("gL\n"); break;
    case EL: printf("EL\n"); break;
    case DT: printf("DT\n"); break;
    case VR: printf("VR\n"); break;
    case VT: printf("VT\n"); break;
    case SPKREFT: printf("spkreft\n"); break;
    case SPKWGAIN: printf("spkwgain\n"); break;

    default: error("Globals::popparan2str","No such popparam_t");

    }

    return "";

}


string Globals::prjparam2str(Prjparam_t prjparam_t) {

    switch (prjparam_t) {

    case PDENS: printf("pdens\n"); break;
    case WDENS: printf("wdens\n"); break;
    case PSILENT: printf("psilent\n"); break;
    case BCPVAR: printf("bcpvar\n"); break;
    case TAUCOND: printf("taucond\n"); break;
    case TAUP: printf("taup\n"); break;
    case TAUB: printf("taub\n"); break;
    case EPS: printf("eps\n"); break;
    case TAUE: printf("taue\n"); break;
    case TAUZI: printf("tauzi\n"); break;
    case TAUZJ: printf("tauzj\n"); break;
    case PRN: printf("prn\n"); break;
    case BGAIN: printf("bgain\n"); break;
    case KBJHALF: printf("kbjhalf\n"); break;
    case EWGAIN: printf("ewgain\n"); break;
    case IWGAIN: printf("iwgain\n"); break;
    case WGAIN: printf("wgain\n"); break;
    case CSPEED: printf("cspeed\n"); break;
    case P0: printf("p0\n"); break;
    case TAUSD: printf("tausd\n"); break;
    case TAUSF: printf("tausf\n"); break;
    case FLPSCR: printf("flpscr\n"); break;
    case PSPA: printf("pspa\n"); break;
    case PSPC: printf("pspc\n"); break;
    case WIJ: printf("wij\n"); break;

    default: error("Globals::prjparam2str","No such prjparam_t");

    }

    return "";

}


string Globals::pop2str(Pop_t pop_t) {

    switch (pop_t) {

    case 0: return "L4Pyr"; break;

    case 1: return "L23Pyr"; break;

    case 2: return "LXDBC"; break;

    case 3: return "LXBC"; break;

    default: error("Globals::pop2str","No such 'pop_t'");

    }

    return "";

}


int Globals::str2pop(string popstr) {

    transform(popstr.begin(),popstr.end(),popstr.begin(),::tolower);

    if (popstr=="l4pyr") return static_cast<int>(Pop_t::L4Pyr);

    if (popstr=="l23pyr") return static_cast<int>(Pop_t::L23Pyr);

    if (popstr=="lxdbc") return static_cast<int>(Pop_t::LXDBC);

    if (popstr=="lxbc") return static_cast<int>(Pop_t::LXBC);

    error("Globals::str2pop","No such 'pop_t'");

    return static_cast<int>(Pop_t::PopN);

}


std::string Globals::prj2str(Prj_t prj_t) {

    switch (prj_t) {

    case 0: return "L4Pyr_L23Pyr"; break;

    case 1: return "L23Pyr_L23Pyr"; break;

    case 2: return "L4Pyr_LXBC"; break;

    case 3: return "L4Pyr_LXDBC"; break; // From stimulated MC to other MC:s in HC

    case 4: return "L23Pyr_LXBC"; break;

    case 5: return "LXBC_L4Pyr"; break;

    case 6: return "LXBC_L23Pyr"; break;

    case 7: return "LXDBC_L4Pyr"; break;

    case 8: return "LXDBC_L23Pyr"; break;

    case 9: return "L23Pyr_L23Pyr_CC"; break;

    case 10: return "L23Pyr_LXDBC_CC"; break;

    case 11: return "L23Pyr_L4Pyr_FF"; break;

    case 12: return "L23Pyr_LXDBC_FF"; break;

    default: error("Globals::prj2str","No such 'prj_t'");

    }

    return "";

}


int Globals::str2prj(string prjstr) {

    transform(prjstr.begin(),prjstr.end(),prjstr.begin(),::tolower);

    if (prjstr=="l4pyr_l23pyr") return static_cast<int>(Prj_t::L4Pyr_L23Pyr);

    if (prjstr=="l4pyr_lxdbc") return static_cast<int>(Prj_t::L4Pyr_LXDBC);

    if (prjstr=="l23pyr_l23pyr") return static_cast<int>(Prj_t::L23Pyr_L23Pyr);

    if (prjstr=="l4pyr_lxbc") return static_cast<int>(Prj_t::L4Pyr_LXBC);

    if (prjstr=="l23pyr_lxbc") return static_cast<int>(Prj_t::L23Pyr_LXBC);

    if (prjstr=="lxbc_l4pyr") return static_cast<int>(Prj_t::LXBC_L4Pyr);

    if (prjstr=="lxbc_l23pyr") return static_cast<int>(Prj_t::LXBC_L23Pyr);

    if (prjstr=="lxdbc_l4pyr") return static_cast<int>(Prj_t::LXDBC_L4Pyr);

    if (prjstr=="lxdbc_l23pyr") return static_cast<int>(Prj_t::LXDBC_L23Pyr);

    if (prjstr=="l23pyr_l23pyr_cc") return static_cast<int>(Prj_t::L23Pyr_L23Pyr_CC);

    if (prjstr=="l23pyr_lxdbc_cc") return static_cast<int>(Prj_t::L23Pyr_LXDBC_CC);

    if (prjstr=="l23pyr_l4pyr_ff") return static_cast<int>(Prj_t::L23Pyr_L4Pyr_FF);

    if (prjstr=="l23pyr_lxdbc_ff") return static_cast<int>(Prj_t::L23Pyr_LXDBC_FF);

    error("Globals::str2prj","No such 'prj_t'");

    return static_cast<int>(Prj_t::PrjN);

}


void Globals::resetstateall() {

    Pop::resetstateall();
    Prj::resetstateall();

    Prj::updateall();
    Pop::updateall();

}

void Globals::simulate(int nstep,bool dolog, bool extended_simulate) {

    setupdone = true;

    for (int step=0; step<nstep; step++) {

    Prj::updateall();

    Pop::updateall();


    if (dolog) Logger::dologall();

    if (!extended_simulate) // When you do not want to have it behave like a single extendded simulation (can chain multiple simulate statements as if one statement)
        Pop::resetbwsupall();


    simstep++;

    }
}

// hexgrid code adapted from ../../misc/Hecgrid.py

int Globals::shell1n(int nshell) {

    if (nshell==0) return 1;

    return 6 * nshell;

}


int Globals::shelln(int nshell) {

    int n = 1;

    for (int sh=1; sh<nshell+1; sh++) n += shell1n(sh);

    return n;

}


int Globals::nshell(int minn) {

    if (minn==0) error("Globals::nshell","Illegal minn<=0");

    int nsh = 0,n = 1;

    while (n<minn) { nsh++; n = shelln(nsh); }

    return nsh;

}


vector<Point_2D>  Globals::hexgrid1(int nshell,float spc,Point_2D origin,
				    float scale,vector<Point_2D> coord2Ds) {

    Point_2D coord2D;

    float yspc = spc * sqrt(3)/2,x0 = origin.x,y0 = origin.y;

    for (int j=0; j<nshell + 1; j++) {

	for (int i=-nshell; i<nshell + 1 - j; i++) {

	    coord2D.x = x0 + (i + j/2.) * spc;

	    coord2D.y = y0 + j * yspc;

	    coord2Ds.push_back(coord2D);

	    if (j>0) {

		coord2D.y = y0 - j * yspc;

		coord2Ds.push_back(coord2D);

	    }
	}
    }

    return coord2Ds;

}

Point_2D Globals::hexgrid1coord(int K,int nshell,float spc,Point_2D origin,float scale) {

    Point_2D coord2D;

    float yspc = spc * sqrt(3)/2,x0 = origin.x,y0 = origin.y;

    K += 1;

    int k = 1;

    for (int j=0; j<nshell + 1; j++) {

	for (int i=-nshell; i<nshell + 1 - j; i++) {

	    if (K>0 and k>K) break;

	    coord2D.x = x0 + (i + j/2.) * spc;

	    coord2D.y = y0 + j * yspc;

	    if (j>0 and (k<0 or k<K)) {

		coord2D.y = y0 - j * yspc;

		k++;

	    }

	    k++;

	}
    }

    return coord2D;

}


vector<vector<Point_2D> > Globals::hexgrid2(int nshell1,float spc1,int nshell2,float spc2,Point_2D origin,
				   float scale) {

    if (spc2==0) spc2 = spc1/(2*nshell2 + 1);

    vector<Point_2D> coord2D_c = hexgrid1(nshell1,spc1,origin,scale);

    vector<vector<Point_2D> > coord2D_h;

    for (size_t c1=0; c1<coord2D_c.size(); c1++)

    	coord2D_h.push_back(hexgrid1(nshell2,spc2,coord2D_c[c1],scale));

    return coord2D_h;

}


std::vector<std::vector<Point_2D> > Globals::hexgridhm(int H,int M,float spcm) {

    int nshellh = nshell(H),nshellm = nshell(M);

    return hexgrid2(nshellh,(2*nshellm + 1)*spcm,nshellm);

}
