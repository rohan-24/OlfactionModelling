#ifndef __Globals_included
#define __Globals_included

#include <vector>
#include <string>
#include <random>
#include <iterator>
#include <algorithm>
#include <iostream>
#include <sys/time.h> // std::clock_gettime
#include <iomanip>      // std::setprecision


namespace Globals {
    
    extern std::string BCPNNSim_version;

    extern bool initialized,setupdone;

    extern float timestep;

    extern struct timespec start; 

    extern int simstep ;

    extern float maxfloat,minfloat,epsfloat;

    extern long gseed;

    extern std::mt19937_64 generator;

    extern std::uniform_real_distribution<float> uniformfloatdistr;

    extern std::uniform_int_distribution<int> uniformintdistr;

    extern double updbwtime;

    extern double proptime;

    extern double hcupdtime;

    extern std::vector<std::vector<float> > trdata,tedata,trlbl,telbl;

    enum Geom_t { RND2D = 0, REC2D, HEX2D  } ;

    enum Pat_t { ORTHO = 0, RND01, RND, NOHRND01 } ;

    enum Actfn_t { NOACTFN = 0, BCP, LIN, LOG, EXP, SIG, SPKBCP, SPKLIN, SPKLOG, SPKEXP, SPKSIG,
		   AREG, ALIF, AdEx, AdExS } ;

    enum Pupdfn_t { ORIGIN = 0, PNORMAL} ;

    enum Normfn_t { NONORMFN = 0, CAP, HALF, FULL } ;

    enum BCPvar_t { FIXED = 0, COUNT, INCR, INCR_E } ;

    enum Prjarch_t { HDIAG = 0, HNDIAG, HPATCHY, UDIAG, UNDIAG, HDIAG_UNDIAG, USELONE, UEXCLONE, BC_MC } ;

    enum Pop_t { L4Pyr = 0, L23Pyr, LXDBC, LXBC, PopN } ;

    enum Prj_t { L4Pyr_L23Pyr = 0,
		 L23Pyr_L23Pyr,
    		 L4Pyr_LXBC,
		 L4Pyr_LXDBC,
		 L23Pyr_LXBC,
		 LXBC_L4Pyr,
		 LXBC_L23Pyr,
    		 LXDBC_L4Pyr,
		 LXDBC_L23Pyr,
		 L23Pyr_L23Pyr_CC,
		 L23Pyr_LXDBC_CC,
		 L23Pyr_L4Pyr_FF,
		 L23Pyr_LXDBC_FF,
		 PrjN } ;

    enum Popparam_t {TAUM = 0, IGAIN, IBIAS, AGAIN, BWGAIN, NAMP, NMEAN, ADGAIN, TAUA, MAXFQ, THRES, SPKPER,
		     SPKPHA, ESYN, CM, GL, EL, DT, VR, VT, SPKREFT, SPKWGAIN } ;

    enum Prjparam_t {PDENS = 0, WDENS, PSILENT, BCPVAR, TAUCOND, TAUP, TAUB, EPS, TAUE, TAUZI, TAUZJ, PRN,
		     BGAIN, KBJHALF, EWGAIN, IWGAIN, WGAIN, CSPEED, P0,TAUSD, TAUSF,FLPSCR, PSPA,
		     PSPC, MINAGE, WIJ } ;

    void ginitialize(bool printversion = true) ;

    void error(std::string errloc,std::string errstr) ;

    void warning(std::string warnloc,std::string warnstr) ;

    void startnanosectimer() ;

    long getnanosec() ;

    long gsetseed(long newgseed = 0) ;

    long ggetseed() ;

    float gnextfloat() ;

    int gnextint() ;

    float distance(std::vector<float> xyz1,std::vector<float> xyz2) ;

    float calctaudt1(float tau) ;

    float calctaudt(float tau) ;

    std::vector<std::vector<float> > mkpats(int Z,int H,int M,Pat_t pattype = ORTHO,float p1 = 1,float p0 = -1,
					    int verbosity = 0) ;

    void cpyvectovec(std::vector<float> frvec,std::vector<float> &tovec,int ioffs = 0) ;

    void cpyrectomat(std::vector<std::vector<float> > &mat,std::vector<std::vector<float> > rec,int r0,int c0) ;

    void cpymattorec(std::vector<std::vector<float> > mat,std::vector<std::vector<float> > &rec,int r0,int c0) ;
   
    std::vector<std::vector<float> > readpats(int plen,std::string filename) ;

    std::vector<std::vector<float> > readpats(int npat,int plen,std::string filename) ;

    std::vector<std::vector<float> > readtxtpats(std::string filename) ;

    std::vector<std::vector<int> > readtxtipats(std::string filename) ;

    std::string actfn2str(Globals::Actfn_t actfn_t) ;

    std::string normfn2str(Globals::Normfn_t normfn_t) ;

    std::string popparam2str(Popparam_t popparam_t) ;

    std::string prjparam2str(Prjparam_t prjparam_t) ;

    std::string pop2str(Pop_t pop_t) ;

    int str2pop(std::string popstr) ;

    std::string prj2str(Prj_t prj_t) ;

    int str2prj(std::string prjstr) ;

    void resetstateall() ;

    void simulate(int nstep = 1,bool dolog = true, bool extended_simulate = false) ;

    struct Point_2D {

	float x,y;

    } ;

    int shell1n(int nshell) ;

    int shelln(int nshell) ;

    int nshell(int minn = 0) ;
 
    std::vector<Point_2D> hexgrid1(int nshell,float spc,Point_2D origin = Point_2D{0,0},float scale = 1,
				   std::vector<Point_2D> coord2Ds = std::vector<Point_2D>(0)) ;

    Point_2D hexgrid1coord(int K,int nshell,float spc,Point_2D origin,float scale) ;

    std::vector<std::vector<Point_2D> > hexgrid2(int nshell1,float spc1,int nshell2,float spc2 = 0,
						 Point_2D origin = Point_2D{0,0},float scale = 1) ;

    std::vector<std::vector<Point_2D> > hexgridhm(int H,int M,float spcm = 5e-4) ;


    /******** Template section ********/

    template<typename T> bool assertnz(T val) { return val!=0; }

    template<typename T> bool assertgte(T val,T valmin = 0) { return valmin<=val; }

    template<typename T> T assertgt(T val,T valmin = 0) { return valmin<val; }

    template<typename T> T assertlte(T val,T valmax = 0) { return val<=valmax; }

    template<typename T> T assertlt(T val,T valmax = 0) { return val<valmax; }

    template<typename T> T assertgtelt(T val,T valmin = 0,T valmax = 1) {

	return assertgte(val,valmin) and assertlt(val,valmax);

    }

    template<typename T> void prnval(T val,int ndec = -1,int nlen = -1,bool endl = true) {

	std::cout << std::setw(nlen)
		  << std::setfill(' ')
		  << std::setprecision(ndec)
		  << std::fixed 
		  << val << " ";
	if (endl) std::cout << std::endl;

    }

    template<typename T>
	void prnvec(const T& t,int ndec = -1,int nlen = -1,bool endl = true) {

	for (size_t i=0; i<t.size(); i++) prnval(t[i],ndec,nlen,false);

	if (endl) std::cout << std::endl;
	
    }

    template<typename T>
	void prnmat(const T& t,int ndec = -1,int nlen = -1,bool endl = true) {

	for (size_t r=0; r<t.size(); r++) { prnvec(t[r],ndec,nlen,true); fflush(stdout); }

	if (endl) std::cout << std::endl;
	
    }    

    template<typename T> void prneval(T val,int ndec = -1,int nlen = -1,bool endl = true) {

	std::cout << std::setw(nlen)
		  << std::setfill(' ')
		  << std::setprecision(ndec)
		  << std::scientific
		  << val << " ";
	if (endl) std::cout << std::endl;

    }

    template<typename T>
	void prnevec(const T& t,int ndec = -1,int nlen = -1,bool endl = true) {

	for (size_t i=0; i<t.size(); i++) prneval(t[i],ndec,nlen,false);

	if (endl) std::cout << std::endl;
	
    }

    template<typename T>
	void prnemat(const T& t,int ndec = -1,int nlen = -1,bool endl = true) {

	for (size_t r=0; r<t.size(); r++) { prnevec(t[r],ndec,nlen,true); fflush(stdout); }

	if (endl) std::cout << std::endl;
	
    }    

    template<typename T>
    void prnval(T *arr,int N,int ndec = -1,int nlen = -1,bool endl = true) {

	for (size_t i=0; i<N; i++) prnval(arr[i],ndec,nlen,false);
	if (endl) std::cout << std::endl;

    }

    template<typename T>
    void fwriteval(T val,std::string filename,std::string mode = "w") {

	FILE *outf = fopen(filename.c_str(),mode.c_str());
	if (outf==NULL) error("gGlobals::writeval","Could not open file");

	fwrite(&val,1,sizeof(T),outf);
	fclose(outf);

    }

    template<typename T>
    void fwriteval(T *val,int N,std::string filename,std::string mode = "w") {

	FILE *outf = fopen(filename.c_str(),mode.c_str());
	if (outf==NULL) error("gGlobals::writeval","Could not open file");
	fwrite(val,N,sizeof(T),outf);
	fclose(outf);

    }

    template<typename T,typename A>
	void fwriteval(std::vector<T,A> const& vec,std::string filename,std::string mode = "w") {

	FILE *outf = fopen(filename.c_str(),mode.c_str());
	if (outf==NULL) error("gGlobals::writeval","Could not open file");

	fwrite(vec.data(),vec.size(),sizeof(T),outf);
	fclose(outf);

    }
    

    template<typename T>
	void fwriteval(T val,FILE *outf) { fwrite(&val,1,sizeof(T),outf); }

    template<typename T>
	void fwriteval(T *val,int N,FILE *outf) { fwrite(val,N,sizeof(T),outf); }
    
    template<typename T,typename A>
	void fwriteval(std::vector<T,A> const& vec,FILE *outf) {

	fwrite(vec.data(),vec.size(),sizeof(T),outf);

    }
    
    template<typename T>
	void fwritemat(const T& t,FILE *outf) {

	for (size_t r=0; r<t.size(); r++) fwriteval(t[r],outf);

    }

    template<typename T>
    	void fwritemat(const T& t,std::string filename) {

    	FILE *outf = fopen(filename.c_str(),"w");

    	for (size_t r=0; r<t.size(); r++) fwriteval(t[r],outf);

    	fclose(outf);

    }
    

    template<typename T>
	int argmax(const std::vector<T>& v,int i0,int i1) {

	T max = v[i0];
	int imax = i0;

	for (size_t i=i0+1; i<i1; i++)

	    if (v[i]>max) { imax = i; max = v[i]; }

	return imax;
	
    }


}

#endif // __Globals_included
