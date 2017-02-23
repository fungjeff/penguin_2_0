//=================================================================================

__device__ void riemann( int &n, int nmax, StateVar &S, sdp *umid, sdp *pmid, bool &riemann_success){

// Solve the Riemann shock tube problem for the left and right input states,
// using the Newton interation procedure described in van Leer (1979).
//---------------------------------------------------------------------------------

  #if EOS>0
  int lim = 10;
  sdp  en_l, en_r, dum, cs_l, cs_r, uml, umr;
  #endif
  sdp pm, pl, rl, ul, pr, rr, ur, t1, t2;

  riemann_success = true;
  if (n>=3 && n<nmax-2)
  {
    riemann_success = false;
    pl   = S.pl;
    rl   = S.rl;
    ul   = S.ul;
    pr   = S.pr;
    rr   = S.rr;
    ur   = S.ur;

    #if EOS == 0
/*
    rl   = csqrt(rl);
    rr   = csqrt(rr);

    t1 = (ur - ul) * rl * rr / (rl + rr);
    t2 = ( pr*rl + pl*rr ) / (rl + rr);

    pm = -0.5*t1 + csqrt(0.25*t1*t1 + t2);

    t1 = pm*rl;
    t2 = pm*rr;
    pm = pm*pm;
    riemann_success = true;

    uml = ul - (pm - pl) / t1;
    umr = ur + (pm - pr) / t2;
    umid[n] = 0.5*(uml + umr);
    pmid[n] = cmax(smallp,pm);
*/
    t1 = 0.5 * (csqrt(pl/rl) + csqrt(pr/rr));
    t2 =-0.5 * (ur - ul) / t1;

    pm = csqrt(pl*pr) * cexp(t2);

    umid[n] = ul - t1 * clog(pm/pl);
    pmid[n] = cmax(smallp,pm);
    riemann_success = true;

    #elif EOS == 2
    
    cs_l = csqrt(gam*pl/rl);
    cs_r = csqrt(gam*pr/rr);
    //t1   = cpow(pl/pr,gamz);
    //uml  = (t1*ul/cs_l + ur/cs_r + 2.0*(t1-1.0)/gamm) / (t1/cs_l + 1.0/cs_r);
    //pm   = 0.5*( pl * cpow(1.0 + gamm*(ul-uml)/2.0/cs_l, 1.0/gamz) + pr * cpow(1.0 + gamm*(uml-ur)/2.0/cs_r, 1.0/gamz) );
    //umid[n] = uml;

    pm = cpow( (cs_l+cs_r - (gamm/2.0)*(ur-ul)) / (cs_l*cpow(pl,-gamz) + cs_r*cpow(pr,-gamz)) , 1.0/gamz);
    if (pm>pr) t1 = (pm-pr) * csqrt(2.0/(rr*(gamfac2*pm + pr*gamm)));
    else       t1 = 2.0*cs_r*(cpow(pm/pr ,gamz)-1.0)/gamm;
    if (pm>pl) t2 = (pm-pl) * csqrt(2.0/(rl*(gamfac2*pm + pl*gamm)));
    else       t2 = 2.0*cs_l*(cpow(pm/pl ,gamz)-1.0)/gamm;

    umid[n] = 0.5*(ur+ul)+0.5*(t1-t2);
    pmid[n] = cmax(smallp,pm);

    #else

    cs_l = csqrt(gam*pl*rl);
    cs_r = csqrt(gam*pr*rr);
    rl   = 1.0/rl;
    rr   = 1.0/rr;
    pm   = pr - pl - cs_r*(ur-ul);
    pm   = pl + pm * cs_l/(cs_l+cs_r);
    pm   = cmax(smallp,pm);
    for (int l=0; l<lim; l++)
    {
      dum  = pm;
      t1   = 1.0 + gamfac1*(pm - pl) / pl;
      t2   = 1.0 + gamfac1*(pm - pr) / pr;
      t1   = cs_l * csqrt(t1);
      t2   = cs_r * csqrt(t2);  
      en_l = 4.0 * rl * t1 * t1;
      en_r = 4.0 * rr * t2 * t2;
      en_l = -en_l * t1/(en_l - gamfac2*(pm - pl));
      en_r =  en_r * t2/(en_r - gamfac2*(pm - pr));
      uml  = ul - (pm - pl) / t1;
      umr  = ur + (pm - pr) / t2;
      pm   = pm + (umr - uml)*(en_r * en_l) / (en_r - en_l);
      pm   = cmax(smallp,pm);

      if (cabs(pm-dum)/pm < 1.0e-8)
      {
        riemann_success = true;
        l=lim;
      }
    }

    uml = ul - (pm - pl) / t1;
    umr = ur + (pm - pr) / t2;
    umid[n] = 0.5*(uml + umr);
    pmid[n] = cmax(smallp,pm);

    #endif
  }
  __syncthreads();

  return;
}
