#ifndef __HCU_included
#define __HCU_included

#include <vector>
#include <string>
#include <random>

#include "Globals.h"

#include "Pop.h"

class Prj;

class HCU : public Pop {

 protected:

    static std::vector<HCU *> hcus;

    std::vector<float> expdsup;
    float expdsupsum,dsupmax;

    friend class BCU;
    friend class Prj;

 public:

    HCU(int M,Globals::Actfn_t actfn_t = Globals::BCP,Globals::Normfn_t normfn = Globals::FULL) ;

    static HCU *gethcu(int id) ;

    void allocatestate() ;

    int getinfo(std::string infostr) ;

    float getstate1(std::string statestr) ;

    virtual std::vector<float> const &getstate(std::string statestr) ;

    void prnstate(std::string statestr) ;

    void fwritestate(std::string statestr,FILE *outf) ;

    virtual Logger *logstate(std::string statestr,std::string filename,int logstep = 1) ;

    virtual void setinp(float inpval,int n = -1) ;

    virtual void setinp(std::vector<float> inpvec) ;

    virtual void resetstate() ;

    void upddsupmax() ;

    void updexpdsup() ;

    void updexpdsupsum() ;

    void updBCP() ;

    virtual void normalize() ;

    virtual void update() ;

} ;

#endif // __HCU_included

