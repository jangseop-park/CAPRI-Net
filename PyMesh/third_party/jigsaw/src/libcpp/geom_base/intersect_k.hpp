

    
    //!! TODO:
    //!! should use dd_flt wherever an intersection
    //!!    is interpolated from existing points
    //!!
    //!! should apply a careful interpolation calc.
    //!!    a'la line_tria -- any existing routines
    //!!    that compute a "midpoint" need updating
    //!!



    /*
    --------------------------------------------------------
     * INTERSECT-K: various (robust) intersection tests. 
    --------------------------------------------------------
     *
     * This program may be freely redistributed under the 
     * condition that the copyright notices (including this 
     * entire header) are not removed, and no compensation 
     * is received through use of the software.  Private, 
     * research, and institutional use is free.  You may 
     * distribute modified versions of this code UNDER THE 
     * CONDITION THAT THIS CODE AND ANY MODIFICATIONS MADE 
     * TO IT IN THE SAME FILE REMAIN UNDER COPYRIGHT OF THE 
     * ORIGINAL AUTHOR, BOTH SOURCE AND OBJECT CODE ARE 
     * MADE FREELY AVAILABLE WITHOUT CHARGE, AND CLEAR 
     * NOTICE IS GIVEN OF THE MODIFICATIONS.  Distribution 
     * of this code as part of a commercial system is 
     * permissible ONLY BY DIRECT ARRANGEMENT WITH THE 
     * AUTHOR.  (If you are not directly supplying this 
     * code to a customer, and you are instead telling them 
     * how they can obtain it for free, then you are not 
     * required to make any arrangement with me.) 
     *
     * Disclaimer:  Neither I nor: Columbia University, The
     * Massachusetts Institute of Technology, The 
     * University of Sydney, nor The National Aeronautics
     * and Space Administration warrant this code in any 
     * way whatsoever.  This code is provided "as-is" to be 
     * used at your own risk.
     *
    --------------------------------------------------------
     *
     * Last updated: 07 January, 2019
     *
     * Copyright 2013-2019
     * Darren Engwirda
     * de2363@columbia.edu
     * https://github.com/dengwirda/
     *
    --------------------------------------------------------
     */

#   pragma once

#   ifndef __INTERSECT_K__
#   define __INTERSECT_K__

    namespace geometry
    {

    /*
    --------------------------------------------------------
     * helper: solve quadratic equations
    --------------------------------------------------------
     */

    template <
    typename      data_type
             >
    __normal_call bool_type polyroots (
        data_type _aa,      // aa*xx^2+bb*xx+cc=0
        data_type _bb,
        data_type _cc,
    __write_ptr  (data_type) _xx
        )
    {
        bool_type _real = false ;

        data_type _sq = _bb * _bb -
       (data_type)+4. * _aa * _cc ;

        if (_sq >= (data_type)+0.)  // real roots
        {
            _sq  = std::sqrt(_sq) ;
        
            _real = true;

            _xx[0] = (-_bb + _sq) ;
            _xx[1] = (-_bb - _sq) ;

            data_type _xm = std::max (
                std::abs(_xx[0]), 
                std::abs(_xx[1])) ;

            data_type _rt = 
        +std::numeric_limits<data_type>::epsilon();

            if (_aa >= _xm * _rt)
            {
                _aa *=(data_type)+2.;

                _xx[0] /= _aa ;
                _xx[1] /= _aa ;
            }
            else
            {
                _xx[0] = -_cc / _bb ;
                _xx[1] = -_cc / _bb ;
            }
        }

        return _real ;
    }

    /*
    --------------------------------------------------------
     * BALL-LINE-KD: ball-line intersections
    --------------------------------------------------------
     */

    template <
    typename      data_type
             >
    __normal_call size_t ball_line_2d (
    __const_ptr  (data_type) _pc, // ball BB(pc,rc)
        data_type            _rc,
    __const_ptr  (data_type) _pa, // line
    __const_ptr  (data_type) _pb,
    __write_ptr  (data_type) _qa, // intersections
    __write_ptr  (data_type) _qb
        )
    {
        size_t _nn = +0;
        
        data_type _pm[2] = {
       (data_type)+.5 * (_pa[0]+_pb[0]),
       (data_type)+.5 * (_pa[1]+_pb[1])
            } ;
        data_type _pd[2] = {
       (data_type)+.5 * (_pb[0]-_pa[0]),
       (data_type)+.5 * (_pb[1]-_pa[1])
            } ;
        data_type _mc[2] = {
       (data_type)+1. * (_pm[0]-_pc[0]),
       (data_type)+1. * (_pm[1]-_pc[1])
            } ;
        
        data_type _aa = dot_2d(_pd, _pd) ;
        data_type _bb = dot_2d(_pd, _mc) *
                       (data_type) +2. ;    
        data_type _cc = dot_2d(_mc, _mc) ;
        _cc -= _rc * _rc ;
        
        data_type _tt[2] ;
        if (polyroots(_aa, _bb, _cc, _tt))
        {
        if (_tt[0] >= (data_type)-1. &&
            _tt[0] <= (data_type)+1. )
        {
            data_type *_qq = _nn++ == +0 
                           ? _qa 
                           : _qb ;
            
            dd_flt _WB = _tt[0] ;
            _WB = (dd_flt)+1. + _WB;
            _WB = (dd_flt)+.5 * _WB;
            
            dd_flt _WA = _tt[0] ;
            _WA = (dd_flt)+1. - _WA;
            _WA = (dd_flt)+.5 * _WA;
            
            dd_flt _PA[2] ;
            _PA[0]=_pa[0] ;
            _PA[1]=_pa[1] ;
            
            dd_flt _PB[2] ;
            _PB[0]=_pb[0] ;
            _PB[1]=_pb[1] ;
            
            dd_flt _QQ[2] ;
            _QQ[0]=_PA[0] * _WA +
                   _PB[0] * _WB ;
            _QQ[1]=_PA[1] * _WA +
                   _PB[1] * _WB ;
                   
            _qq[0]=_QQ[0] ;
            _qq[1]=_QQ[1] ;
        }

        if (_tt[1] >= (data_type)-1. &&
            _tt[1] <= (data_type)+1. )
        {
            data_type *_qq = _nn++ == +0  
                           ? _qa 
                           : _qb ;
                           
            dd_flt _WB = _tt[1] ;
            _WB = (dd_flt)+1. + _WB;
            _WB = (dd_flt)+.5 * _WB;
            
            dd_flt _WA = _tt[1] ;
            _WA = (dd_flt)+1. - _WA;
            _WA = (dd_flt)+.5 * _WA;
            
            dd_flt _PA[2] ;
            _PA[0]=_pa[0] ;
            _PA[1]=_pa[1] ;
            
            dd_flt _PB[2] ;
            _PB[0]=_pb[0] ;
            _PB[1]=_pb[1] ;
            
            dd_flt _QQ[2] ;
            _QQ[0]=_PA[0] * _WA +
                   _PB[0] * _WB ;
            _QQ[1]=_PA[1] * _WA +
                   _PB[1] * _WB ;
                   
            _qq[0]=_QQ[0] ;
            _qq[1]=_QQ[1] ;
        }
        }

        return ( _nn ) ; // return num roots
    }

    template <
    typename      data_type
             >
    __normal_call size_t ball_line_3d (
    __const_ptr  (data_type) _pc, // ball BB(pc,rc)
        data_type            _rc,
    __const_ptr  (data_type) _pa, // line
    __const_ptr  (data_type) _pb,
    __write_ptr  (data_type) _qa, // intersections
    __write_ptr  (data_type) _qb
        )
    {
        size_t _nn = +0;
        
        data_type _pm[3] = {
       (data_type)+.5 * (_pa[0]+_pb[0]),
       (data_type)+.5 * (_pa[1]+_pb[1]),
       (data_type)+.5 * (_pa[2]+_pb[2])
            } ;
        data_type _pd[3] = {
       (data_type)+.5 * (_pb[0]-_pa[0]),
       (data_type)+.5 * (_pb[1]-_pa[1]),
       (data_type)+.5 * (_pb[2]-_pa[2])
            } ;
        data_type _mc[3] = {
       (data_type)+1. * (_pm[0]-_pc[0]),
       (data_type)+1. * (_pm[1]-_pc[1]),
       (data_type)+1. * (_pm[2]-_pc[2])
            } ;
        
        data_type _aa = dot_3d(_pd, _pd) ;
        data_type _bb = dot_3d(_pd, _mc) *
                       (data_type) +2. ;    
        data_type _cc = dot_3d(_mc, _mc) ;
        _cc -= _rc * _rc ;
        
        data_type _tt[2] ;
        if (polyroots(_aa, _bb, _cc, _tt))
        {
        if (_tt[0] >= (data_type)-1. &&
            _tt[0] <= (data_type)+1. )
        {
            data_type *_qq = _nn++ == +0 
                           ? _qa 
                           : _qb ;
             
            dd_flt _WB = _tt[0] ;
            _WB = (dd_flt)+1. + _WB;
            _WB = (dd_flt)+.5 * _WB;
            
            dd_flt _WA = _tt[0] ;
            _WA = (dd_flt)+1. - _WA;
            _WA = (dd_flt)+.5 * _WA;
            
            dd_flt _PA[3] ;
            _PA[0]=_pa[0] ;
            _PA[1]=_pa[1] ;
            _PA[2]=_pa[2] ;
            
            dd_flt _PB[3] ;
            _PB[0]=_pb[0] ;
            _PB[1]=_pb[1] ;
            _PB[2]=_pb[2] ;
            
            dd_flt _QQ[3] ;
            _QQ[0]=_PA[0] * _WA +
                   _PB[0] * _WB ;
            _QQ[1]=_PA[1] * _WA +
                   _PB[1] * _WB ;
            _QQ[2]=_PA[2] * _WA +
                   _PB[2] * _WB ; 
                   
            _qq[0]=_QQ[0] ;
            _qq[1]=_QQ[1] ;
            _qq[2]=_QQ[2] ;
        }

        if (_tt[1] >= (data_type)-1. &&
            _tt[1] <= (data_type)+1. )
        {
            data_type *_qq = _nn++ == +0  
                           ? _qa 
                           : _qb ;
           
            dd_flt _WB = _tt[1] ;
            _WB = (dd_flt)+1. + _WB;
            _WB = (dd_flt)+.5 * _WB;
            
            dd_flt _WA = _tt[1] ;
            _WA = (dd_flt)+1. - _WA;
            _WA = (dd_flt)+.5 * _WA;
            
            dd_flt _PA[3] ;
            _PA[0]=_pa[0] ;
            _PA[1]=_pa[1] ;
            _PA[2]=_pa[2] ;
            
            dd_flt _PB[3] ;
            _PB[0]=_pb[0] ;
            _PB[1]=_pb[1] ;
            _PB[2]=_pb[2] ;
            
            dd_flt _QQ[3] ;
            _QQ[0]=_PA[0] * _WA +
                   _PB[0] * _WB ;
            _QQ[1]=_PA[1] * _WA +
                   _PB[1] * _WB ;
            _QQ[2]=_PA[2] * _WA +
                   _PB[2] * _WB ; 
                   
            _qq[0]=_QQ[0] ;
            _qq[1]=_QQ[1] ;
            _qq[2]=_QQ[2] ;
        }
        }

        return ( _nn ) ; // return num roots
    }

    template <
    typename      data_type
             >
    __normal_call bool line_flat_3d (
    __const_ptr  (data_type) _pp, // (xx-pp).nv=0
    __const_ptr  (data_type) _nv,
    __const_ptr  (data_type) _pa, // line
    __const_ptr  (data_type) _pb,
    __write_ptr  (data_type) _qq, // intersection
        bool _bind =   true
        )
    {
        if (_pa[0] == _pb[0] &&
            _pa[1] == _pb[1] &&
            _pa[2] == _pb[2] )
        {
          //std::cout << "node_flat_3d" << std::endl;

            return ( false ) ;
        }
        else
        if (_pp[0] == _pa[0] &&
            _pp[1] == _pa[1] &&
            _pp[2] == _pa[2] )
        {
        
            _qq[0] =  _pa[0] ;
            _qq[1] =  _pa[1] ;
            _qq[2] =  _pa[2] ;

            return (  true ) ;
        }
        else
        if (_pp[0] == _pb[0] &&
            _pp[1] == _pb[1] &&
            _pp[2] == _pb[2] )
        {
        
            _qq[0] =  _pb[0] ;
            _qq[1] =  _pb[1] ;
            _qq[2] =  _pb[2] ;

            return (  true ) ;
        }
        else
        {

        data_type _ab[3];
        _ab[0] = _pb[0] - _pa[0] ;
        _ab[1] = _pb[1] - _pa[1] ;
        _ab[2] = _pb[2] - _pa[2] ;

        data_type _ap[3];
        _ap[0] = _pp[0] - _pa[0] ;
        _ap[1] = _pp[1] - _pa[1] ;
        _ap[2] = _pp[2] - _pa[2] ;

        data_type _ep = +100. * 
        std::numeric_limits<data_type>::epsilon() ;

        data_type _d1 = 
        geometry::dot_3d(_ap, _nv) ; 
        data_type _d2 = 
        geometry::dot_3d(_ab, _nv) ;

        if (std::abs(_d2) <= _ep * std::abs(_d1))
            return ( false ) ;
        
        data_type _tt =  _d1 / _d2 ;
        
        if (_bind)
        {
        if (_tt  < (data_type)+0.)
            return ( false ) ;
        if (_tt  > (data_type)+1.)
            return ( false ) ;
        }

        if (_tt == (data_type)+0.)
        {
        _qq[0] = _pa[0] ;
        _qq[1] = _pa[1] ;
        _qq[2] = _pa[2] ;
        }
        else
        if (_tt == (data_type)+1.)
        {
        _qq[0] = _pb[0] ;
        _qq[1] = _pb[1] ;
        _qq[2] = _pb[2] ;
        }
        else
        {
        dd_flt _AB[3];
        _AB[0] = _pb[0] ;
        _AB[1] = _pb[1] ;
        _AB[2] = _pb[2] ;
        _AB[0]-= _pa[0] ;
        _AB[1]-= _pa[1] ;
        _AB[2]-= _pa[2] ;

        dd_flt _QQ[3];
        _QQ[0] = _pa[0] ;
        _QQ[1] = _pa[1] ;
        _QQ[2] = _pa[2] ;    
        _QQ[0]+= _AB[0] * _tt ;
        _QQ[1]+= _AB[1] * _tt ;
        _QQ[2]+= _AB[2] * _tt ;

        _qq[0] = _QQ[0] ;
        _qq[1] = _QQ[1] ;
        _qq[2] = _QQ[2] ;
        }

        return ( true ) ;

        }
    }

    template <
    typename      data_type
             >
    __normal_call size_t tria_flat_3d (
    __const_ptr  (data_type) _pp, // (xx-pp).nv=0
    __const_ptr  (data_type) _nv,
    __const_ptr  (data_type) _pa, // tria
    __const_ptr  (data_type) _pb,
    __const_ptr  (data_type) _pc,
    __write_ptr  (data_type) _qa, // intersection
    __write_ptr  (data_type) _qb
        )
    {
        size_t _ni = +0;
        
        if (line_flat_3d (
            _pp, _nv, 
            _pa, _pb, 
           (_ni == +0) ? _qa : _qb))
            _ni += +1  ;
        if (line_flat_3d (
            _pp, _nv, 
            _pb, _pc, 
           (_ni == +0) ? _qa : _qb))
            _ni += +1  ;
        if (line_flat_3d (
            _pp, _nv, 
            _pc, _pa, 
           (_ni == +0) ? _qa : _qb))
            _ni += +1  ;

        return ( _ni ) ;
    }

    template <
    typename      real_type
             >
    __normal_call bool proj_line_2d (
    __const_ptr  (real_type) _pp,
    __const_ptr  (real_type) _pa, // node on line
    __const_ptr  (real_type) _va, // vec. of line
                  real_type &_tt
        )
    {
        _tt = (real_type)+0.0 ;
    
        real_type _ap[2];
        _ap[0] = _pp[0] - _pa[0] ;
        _ap[1] = _pp[1] - _pa[1] ;
        
        real_type _ep = +100. * 
        std::numeric_limits<real_type>::epsilon();
        
        real_type _d1 = 
        geometry::dot_2d(_ap, _va) ;
        real_type _d2 = 
        geometry::dot_2d(_va, _va) ;
        
        if (std::abs(_d2) <= _ep * std::abs(_d1) )
        return ( false ) ;

        _tt =  _d1 / _d2 ;
        
        return (  true ) ;        
    }

    template <
    typename      real_type
             >
    __normal_call bool proj_line_3d (
    __const_ptr  (real_type) _pp,
    __const_ptr  (real_type) _pa, // node on line
    __const_ptr  (real_type) _va, // vec. of line
                  real_type &_tt
        )
    {
        _tt = (real_type)+0.0 ;
    
        real_type _ap[3];
        _ap[0] = _pp[0] - _pa[0] ;
        _ap[1] = _pp[1] - _pa[1] ;
        _ap[2] = _pp[2] - _pa[2] ;
        
        real_type _ep = +100. * 
        std::numeric_limits<real_type>::epsilon();
        
        real_type _d1 = 
        geometry::dot_3d(_ap, _va) ;
        real_type _d2 = 
        geometry::dot_3d(_va, _va) ;
        
        if (std::abs(_d2) <= _ep * std::abs(_d1) )
        return ( false ) ;

        _tt =  _d1 / _d2 ;
        
        return (  true ) ;        
    }

    /*
    --------------------------------------------------------
     * linear intersection kernels
    --------------------------------------------------------
     */

    typedef char_type hits_type;

    hits_type null_hits = +0 ;
    hits_type node_hits = +1 ;
    hits_type edge_hits = +2 ;
    hits_type face_hits = +3 ;
    hits_type tria_hits = +4 ;


    __normal_call double cleave2d (
    __const_ptr  (double) _pa,
    __const_ptr  (double) _pb,
    __const_ptr  (double) _pp
        )
    {
        double _ab[2] ;
        _ab[0] = _pb[0]-_pa[0] ;
        _ab[1] = _pb[1]-_pa[1] ; 

        double _ap[2] ;
        _ap[0] = _pp[0]-_pa[0] ;
        _ap[1] = _pp[1]-_pa[1] ;

        double _dp = 
            _ab[0] * _ap[0] + 
            _ab[1] * _ap[1] ;

        return _dp ;
    }

    __normal_call double cleave3d (
    __const_ptr  (double) _pa,
    __const_ptr  (double) _pb,
    __const_ptr  (double) _pp
        )
    {
        double _ab[3] ;
        _ab[0] = _pb[0]-_pa[0] ;
        _ab[1] = _pb[1]-_pa[1] ;
        _ab[2] = _pb[2]-_pa[2] ;

        double _ap[3] ;
        _ap[0] = _pp[0]-_pa[0] ;
        _ap[1] = _pp[1]-_pa[1] ;
        _ap[2] = _pp[2]-_pa[2] ;

        double _dp = 
            _ab[0] * _ap[0] +
            _ab[1] * _ap[1] +
            _ab[2] * _ap[2] ;

        return _dp ;
    }


    /*
    --------------------------------------------------------
     * NODE-NODE-KD: node-node intersections
    --------------------------------------------------------
     */

    template <
        typename  real_type
             >
    __inline_call hits_type node_node_2d (
    __const_ptr  (real_type) _pa, // node
    __const_ptr  (real_type) _pb, // node
    __write_ptr  (real_type) _qq  // intersection
        )
    {
        if (_pa[0] == _pb[0] &&
            _pa[1] == _pb[1] )
        {
    /*----------------------- have node-node intersection */
            _qq[0] =  _pa[0] ;
            _qq[1] =  _pa[1] ;

            return node_hits ;
        }
        else
        {
    /*----------------------- no intersections: null hits */
            return null_hits ;
        }
    }

    template <
        typename  real_type
             >
    __inline_call hits_type node_node_3d (
    __const_ptr  (real_type) _pa, // node
    __const_ptr  (real_type) _pb, // node
    __write_ptr  (real_type) _qq  // intersection
        )
    {
        if (_pa[0] == _pb[0] &&
            _pa[1] == _pb[1] &&
            _pa[2] == _pb[2] )
        {
    /*----------------------- have node-node intersection */
            _qq[0] =  _pa[0] ;
            _qq[1] =  _pa[1] ;
            _qq[2] =  _pa[2] ;

            return node_hits ;
        }
        else
        {
    /*----------------------- no intersections: null hits */
            return null_hits ;
        }
    }

    /*
    --------------------------------------------------------
     * NODE-LINE-KD: node-line intersections
    --------------------------------------------------------
     */

    template <
        typename  real_type
             >
    __normal_call hits_type node_line_2d (
    __const_ptr  (real_type) _pp, // node
    __const_ptr  (real_type) _pa, // line
    __const_ptr  (real_type) _pb,
    __write_ptr  (real_type) _qq, // intersection
                  bool_type _bind = true
        )
    {
        if (_pa[0] == _pb[0] &&
            _pa[1] == _pb[1] )
        {
    /*----------------------- test node-node intersection */
            return 
            node_node_2d(_pa, _pp, _qq);
        }
        else
        {
    /*----------------------- test node-line intersection */
            double _PA[2] ;
            _PA[0] =  _pa[0] ;
            _PA[1] =  _pa[1] ;

            double _PB[2] ;
            _PB[0] =  _pb[0] ;
            _PB[1] =  _pb[1] ;

            double _PP[2] ;
            _PP[0] =  _pp[0] ;
            _PP[1] =  _pp[1] ;

            double _ss = 
                geompred::orient2d (
                (double*)_PA , 
                (double*)_PB , 
                (double*)_PP )   ;

            if (_ss == +0.0)
            {
    /*----------------------- orient w.r.t. line endpoint */
            double _sa = cleave2d (
                (double*)_PA , 
                (double*)_PB , 
                (double*)_PP )   ;

            double _sb = cleave2d (
                (double*)_PB , 
                (double*)_PA , 
                (double*)_PP )   ;

            if (_sa*_sb > +0.0)
            {
    /*----------------------- have node-line intersection */
            _qq[0] =  _pp[0] ;
            _qq[1] =  _pp[1] ;

            return edge_hits ;
            }
            else
            if (_sa == +0.0)
            {
    /*----------------------- have node-node intersection */        
            _qq[0] =  _pa[0] ;
            _qq[1] =  _pa[1] ;

            return node_hits ;
            }
            else
            if (_sb == +0.0)
            {
    /*----------------------- have node-node intersection */        
            _qq[0] =  _pb[0] ;
            _qq[1] =  _pb[1] ;

            return node_hits ;
            }
            else
            if ( !_bind )
            {
    /*----------------------- is unconstrained: edge hits */
            _qq[0] =  _pp[0] ;
            _qq[1] =  _pp[1] ;

            return edge_hits ;
            }
            else
            {
    /*----------------------- no intersections: null hits */
            return null_hits ;
            }

            }
            else
            {
    /*----------------------- no intersections: null hits */
            return null_hits ;
            }
        }
    }

    template <
        typename  real_type
             >
    __normal_call hits_type node_line_3d (
    __const_ptr  (real_type) _pp, // node
    __const_ptr  (real_type) _pa, // line
    __const_ptr  (real_type) _pb,
    __write_ptr  (real_type) _qq, // intersection
                  bool_type _bind = true
        )
    {
        if (_pa[0] == _pb[0] &&
            _pa[1] == _pb[1] &&
            _pa[2] == _pb[2] )
        {
    /*----------------------- test node-node intersection */
            return 
            node_node_3d(_pa, _pp, _qq);
        }
        else
        {
    /*----------------------- test node-line intersection */
            double _PA[3] ;
            _PA[0] =  _pa[0] ;
            _PA[1] =  _pa[1] ;
            _PA[2] =  _pa[2] ;

            double _PB[3] ;
            _PB[0] =  _pb[0] ;
            _PB[1] =  _pb[1] ;
            _PB[2] =  _pb[2] ;

            double _PP[3] ;
            _PP[0] =  _pp[0] ;
            _PP[1] =  _pp[1] ;
            _PP[2] =  _pp[2] ;

    /*----------------------- get orientation in xy-plane */
            double _A1[2] ;
            _A1[0] =  _PA[0] ;
            _A1[1] =  _PA[1] ;

            double _B1[2] ;
            _B1[0] =  _PB[0] ;
            _B1[1] =  _PB[1] ;

            double _P1[2] ;
            _P1[0] =  _PP[0] ;
            _P1[1] =  _PP[1] ;

            double _s1 = 
                geompred::orient2d (
                (double*)_A1 , 
                (double*)_B1 , 
                (double*)_P1 )   ;

    /*----------------------- get orientation in xz-plane */
            double _A2[2] ;
            _A2[0] =  _PA[0] ;
            _A2[1] =  _PA[2] ;

            double _B2[2] ;
            _B2[0] =  _PB[0] ;
            _B2[1] =  _PB[2] ;

            double _P2[2] ;
            _P2[0] =  _PP[0] ;
            _P2[1] =  _PP[2] ;

            double _s2 = 
                geompred::orient2d (
                (double*)_A2 , 
                (double*)_B2 , 
                (double*)_P2 )   ;

    /*----------------------- get orientation in yz-plane */
            double _A3[2] ;
            _A3[0] =  _PA[1] ;
            _A3[1] =  _PA[2] ;

            double _B3[2] ;
            _B3[0] =  _PB[1] ;
            _B3[1] =  _PB[2] ;

            double _P3[2] ;
            _P3[0] =  _PP[1] ;
            _P3[1] =  _PP[2] ;

            double _s3 = 
                geompred::orient2d (
                (double*)_A3 , 
                (double*)_B3 , 
                (double*)_P3 )   ;

    /*----------------------- test intersection hierarchy */
            if (_s1 == +0.0 &&
                _s2 == +0.0 &&
                _s3 == +0.0 )
            {
    /*----------------------- orient w.r.t. line endpoint */
            double _sa = cleave3d (
                (double*)_PA , 
                (double*)_PB , 
                (double*)_PP )   ;

            double _sb = cleave3d (
                (double*)_PB , 
                (double*)_PA , 
                (double*)_PP )   ;

            if (_sa*_sb > +0.0)
            {
    /*----------------------- have node-line intersection */
            _qq[0] =  _pp[0] ;
            _qq[1] =  _pp[1] ;
            _qq[2] =  _pp[2] ;

            return edge_hits ;
            }
            else
            if (_sa == +0.0)
            {
    /*----------------------- have node-node intersection */        
            _qq[0] =  _pa[0] ;
            _qq[1] =  _pa[1] ;
            _qq[2] =  _pa[2] ;

            return node_hits ;
            }
            else
            if (_sb == +0.0)
            {
    /*----------------------- have node-node intersection */        
            _qq[0] =  _pb[0] ;
            _qq[1] =  _pb[1] ;
            _qq[2] =  _pb[2] ;

            return node_hits ;
            }
            else
            if ( !_bind )
            {
    /*----------------------- is unconstrained: edge hits */
            _qq[0] =  _pp[0] ;
            _qq[1] =  _pp[1] ;
            _qq[2] =  _pp[2] ;

            return edge_hits ;
            }
            else
            {
    /*----------------------- no intersections: null hits */
            return null_hits ;
            }

            }
            else
            {
    /*----------------------- no intersections: null hits */
            return null_hits ;
            }
        }
    }

    /*
    --------------------------------------------------------
     * LINE-LINE-KD: line-line intersections
    --------------------------------------------------------
     */

    template <
        typename  real_type
             >
    __normal_call hits_type line_line_2d (
    __const_ptr  (real_type) _pa, // line
    __const_ptr  (real_type) _pb,
    __const_ptr  (real_type) _pc, // line
    __const_ptr  (real_type) _pd,
    __write_ptr  (real_type) _qq, // intersection
                  bool_type _bind = true ,
                  char_type _part = +1
        )
    {
        if (_pa[0] == _pb[0] &&
            _pa[1] == _pb[1] )
        {
    /*----------------------- test node-line intersection */
            return node_line_2d (
                _pa, _pc, _pd, _qq, _bind) ;
        }
        else
        if (_pc[0] == _pd[0] &&
            _pc[1] == _pd[1] )
        {
    /*----------------------- test node-line intersection */
            return node_line_2d (
                _pc, _pa, _pb, _qq, _bind) ;
        }
        else
        {
    /*----------------------- test line-line intersection */
            double _PA[2] ;
            _PA[0] =  _pa[0] ;
            _PA[1] =  _pa[1] ;

            double _PB[2] ;
            _PB[0] =  _pb[0] ;
            _PB[1] =  _pb[1] ;

            double _PC[2] ;
            _PC[0] =  _pc[0] ;
            _PC[1] =  _pc[1] ;

            double _PD[2] ;
            _PD[0] =  _pd[0] ;
            _PD[1] =  _pd[1] ;

    /*----------------------- orient w.r.t. line endpoint */
            double  _sa = 
                geompred::orient2d (
                (double*)_PA , 
                (double*)_PC , 
                (double*)_PD )   ;

            double  _sb = 
                geompred::orient2d (
                (double*)_PB , 
                (double*)_PD , 
                (double*)_PC )   ;

            if (_bind)
            {
    /*----------------------- no intersections: null hits */
            if (_sa * _sb < +0.0 ) 
            return null_hits;
            }

            double  _sc = 
                geompred::orient2d (
                (double*)_PC , 
                (double*)_PA , 
                (double*)_PB )   ;
            
            double  _sd = 
                geompred::orient2d (
                (double*)_PD , 
                (double*)_PB , 
                (double*)_PA )   ;

            if (_bind)
            {
    /*----------------------- no intersections: null hits */
            if (_sc * _sd < +0.0 ) 
            return null_hits;
            }

            if (_sa == +0.0 &&
                _sb == +0.0 )
            {
    /*----------------------- colinear: test for overlaps */

          //std::cout << "line-line-2d" << std::endl;

            return null_hits ;

            }
            else
            if (_sa == +0.0 )
            {
    /*----------------------- have node-line intersection */
            _qq[0] =  _pa[0] ;
            _qq[1] =  _pa[1] ;
            
            return node_hits ;
            }
            else
            if (_sb == +0.0 )
            {
    /*----------------------- have node-line intersection */
            _qq[0] =  _pb[0] ;
            _qq[1] =  _pb[1] ;
            
            return node_hits ;
            }
            else
            if (_sc == +0.0 )
            {
    /*----------------------- have node-line intersection */
            _qq[0] =  _pc[0] ;
            _qq[1] =  _pc[1] ;
            
            return node_hits ;
            }
            else
            if (_sd == +0.0 )
            {
    /*----------------------- have node-line intersection */
            _qq[0] =  _pd[0] ;
            _qq[1] =  _pd[1] ;
            
            return node_hits ;
            }
            else
            {
    /*----------------------- have line-line intersection */
            double _mm [2*2] ;
            _mm[__ij(0,0,2)] = _PB[0]-_PA[0] ;
            _mm[__ij(1,0,2)] = _PB[1]-_PA[1] ;
            _mm[__ij(0,1,2)] = _PC[0]-_PD[0] ;
            _mm[__ij(1,1,2)] = _PC[1]-_PD[1] ;

            double _im [2*2] , _dm;
            inv_2x2(2, _mm, 2, _im, _dm) ;

            double _rv [2*1] ;
            _rv[__ij(0,0,2)] = _PC[0]-_PA[0] ;
            _rv[__ij(1,0,2)] = _PC[1]-_PA[1] ;

            if (_dm == +0.0)
            return null_hits ;

            double _tu = 
            _im[__ij(0,0,2)] * _rv [0] + 
            _im[__ij(0,1,2)] * _rv [1] ;
            double _tv = 
            _im[__ij(1,0,2)] * _rv [0] + 
            _im[__ij(1,1,2)] * _rv [1] ;

            _tu       /= _dm ; 
            _tv       /= _dm ;

            if (_bind)
            {
                _tu = 
            std::min(+1.,std::max(+0., _tu)) ;
                _tv = 
            std::min(+1.,std::max(+0., _tv)) ;
            }

            if (_part == +1)
            {
    /*----------------------- calc. intersection on [a,b] */
                dd_flt _WA = 1. - _tu ;
                dd_flt _WB = 0. + _tu ;

                dd_flt _FA[2] ;
                _FA[0] =  _pa[0] ;
                _FA[1] =  _pa[1] ;

                dd_flt _FB[2] ;
                _FB[0] =  _pb[0] ;
                _FB[1] =  _pb[1] ;

                dd_flt _QQ[2] ;
                _QQ[0] = _FA[0] * _WA +
                         _FB[0] * _WB ;
                _QQ[1] = _FA[1] * _WA +
                         _FB[1] * _WB ;

                _qq[0] =  _QQ[0] ;
                _qq[1] =  _QQ[1] ;
            }
            else
            if (_part == +2)
            {
    /*----------------------- calc. intersection on [c,d] */
                dd_flt _WC = 1. - _tv ;
                dd_flt _WD = 0. + _tv ;

                dd_flt _FC[2] ;
                _FC[0] =  _pc[0] ;
                _FC[1] =  _pc[1] ;

                dd_flt _FD[2] ;
                _FD[0] =  _pd[0] ;
                _FD[1] =  _pd[1] ;

                dd_flt _QQ[2] ;
                _QQ[0] = _FC[0] * _WC +
                         _FD[0] * _WD ;
                _QQ[1] = _FC[1] * _WC +
                         _FD[1] * _WD ;

                _qq[0] =  _QQ[0] ;
                _qq[1] =  _QQ[1] ;
            }
            else
            {
            __assert( false && 
            "line_line_2d: invalid part!!") ;
            }

            return edge_hits ;
            }

        }
    }

    template <
        typename  real_type
             >
    __normal_call hits_type line_line_3d (
    __const_ptr  (real_type) _pa, // line
    __const_ptr  (real_type) _pb,
    __const_ptr  (real_type) _pc, // line
    __const_ptr  (real_type) _pd,
    __write_ptr  (real_type) _qq, // intersection
                  bool_type _bind = true ,
                  char_type _part = +1
        )
    {

        __unreferenced (_part);

        if (_pa[0] == _pb[0] &&
            _pa[1] == _pb[1] &&
            _pa[2] == _pb[2] )
        {
    /*----------------------- test node-line intersection */
            return node_line_3d (
                _pa, _pc, _pd, _qq, _bind) ;
        }
        else
        if (_pc[0] == _pd[0] &&
            _pc[1] == _pd[1] &&
            _pc[2] == _pd[2] )
        {
    /*----------------------- test node-line intersection */
            return node_line_3d (
                _pc, _pa, _pb, _qq, _bind) ;
        }
        else
        {
    /*----------------------- test line-line intersection */
            /*
            double _PA[3] ;
            _PA[0] =  _pa[0] ;
            _PA[1] =  _pa[1] ;
            _PA[2] =  _pa[2] ;

            double _PB[3] ;
            _PB[0] =  _pb[0] ;
            _PB[1] =  _pb[1] ;
            _PB[2] =  _pb[2] ;

            double _PC[3] ;
            _PC[0] =  _pc[0] ;
            _PC[1] =  _pc[1] ;
            _PC[2] =  _pc[2] ;

            double _PD[3] ;
            _PD[0] =  _pd[0] ;
            _PD[1] =  _pd[1] ;
            _PD[2] =  _pd[2] ;
            */

        //!! actual line intersection...

        //!!std::cout << "line-line-3d" << std::endl;

            return null_hits ;

        }
    }

    /*
    --------------------------------------------------------
     * NODE-TRIA-KD: node-tria intersections
    --------------------------------------------------------
     */

    template <
        typename  real_type
             >
    __normal_call hits_type node_tria_2d (
    __const_ptr  (real_type) _pp, // node
    __const_ptr  (real_type) _p1, // tria
    __const_ptr  (real_type) _p2,
    __const_ptr  (real_type) _p3,
    __write_ptr  (real_type) _qq, // intersection
                  bool_type _bind = true
        )
    {
        if (_p1[0] == _p2[0] &&
            _p1[1] == _p2[1] )
        {
    /*----------------------- test node-line intersection */
            return node_line_2d (
                _pp, _p2, _p3, _qq, _bind) ;
        }
        else
        if (_p2[0] == _p3[0] &&
            _p2[1] == _p3[1] )
        {
    /*----------------------- test node-line intersection */
            return node_line_2d (
                _pp, _p3, _p1, _qq, _bind) ;
        }
        else
        if (_p3[0] == _p1[0] &&
            _p3[1] == _p1[1] )
        {
    /*----------------------- test node-line intersection */
            return node_line_2d (
                _pp, _p1, _p2, _qq, _bind) ;
        }
        else
        {
    /*----------------------- test node-tria intersection */
            double _PP[2] ;
            _PP[0] =  _pp[0] ;
            _PP[1] =  _pp[1] ;

            double _P1[2] ;
            _P1[0] =  _p1[0] ;
            _P1[1] =  _p1[1] ;

            double _P2[2] ;
            _P2[0] =  _p2[0] ;
            _P2[1] =  _p2[1] ;

            double _P3[2] ;
            _P3[0] =  _p3[0] ;
            _P3[1] =  _p3[1] ;

    /*----------------------- orient w.r.t. tria vertices */
            double _s3 = 
                geompred::orient2d (
                (double*)_P1 , 
                (double*)_P2 , 
                (double*)_PP )   ;

            double _s1 = 
                geompred::orient2d (
                (double*)_P2 , 
                (double*)_P3 , 
                (double*)_PP )   ;

            double _s2 = 
                geompred::orient2d (
                (double*)_P3 , 
                (double*)_P1 , 
                (double*)_PP )   ;

    /*----------------------- test intersection hierarchy */
            if (_s1*_s2<0.0 ||
                _s1*_s3<0.0 ||
                _s2*_s3<0.0 )
            {
    /*----------------------- no intersections: null hits */
            return null_hits ;
            }
            else
            if (_s1 == +0.0 &&
                _s2 == +0.0 &&
                _s3 == +0.0 )
            {
    /*----------------------- degenerate tria: check line */

        //!!std::cout << "node-tria-2d" << std::endl;

            return null_hits ;

            }
            else
            if (_s1 == +0.0 &&
                _s2 == +0.0 )
            {
    /*----------------------- have node-node intersection */
            _qq[0] =  _pp[0] ;
            _qq[1] =  _pp[1] ;

            return node_hits ;
            }
            else
            if (_s2 == +0.0 &&
                _s3 == +0.0 )
            {
    /*----------------------- have node-node intersection */
            _qq[0] =  _pp[0] ;
            _qq[1] =  _pp[1] ;

            return node_hits ;
            }
            else
            if (_s3 == +0.0 &&
                _s1 == +0.0 )
            {
    /*----------------------- have node-node intersection */
            _qq[0] =  _pp[0] ;
            _qq[1] =  _pp[1] ;

            return node_hits ;
            }
            else
            if (_s1 == +0.0 )
            {
    /*----------------------- have node-edge intersection */
            _qq[0] =  _pp[0] ;
            _qq[1] =  _pp[1] ;

            return edge_hits ;
            }
            else
            if (_s2 == +0.0 )
            {
    /*----------------------- have node-edge intersection */
            _qq[0] =  _pp[0] ;
            _qq[1] =  _pp[1] ;

            return edge_hits ;
            }
            else
            if (_s3 == +0.0 )
            {
    /*----------------------- have node-edge intersection */
            _qq[0] =  _pp[0] ;
            _qq[1] =  _pp[1] ;

            return edge_hits ;
            }
            else
            {
    /*----------------------- have node-tria intersection */
            _qq[0] =  _pp[0] ;
            _qq[1] =  _pp[1] ;

            return face_hits ;
            }
        }
    }

    template <
        typename  real_type
             >
    __normal_call hits_type node_tria_3d (
    __const_ptr  (real_type) _pp, // node
    __const_ptr  (real_type) _p1, // tria
    __const_ptr  (real_type) _p2,
    __const_ptr  (real_type) _p3,
    __write_ptr  (real_type) _qq, // intersection
                  bool_type _bind = true
        )
    {
        if (_p1[0] == _p2[0] &&
            _p1[1] == _p2[1] &&
            _p1[2] == _p2[2] )
        {
    /*----------------------- test node-line intersection */
            return node_line_3d (
                _pp, _p2, _p3, _qq, _bind) ;
        }
        else
        if (_p2[0] == _p3[0] &&
            _p2[1] == _p3[1] &&
            _p2[2] == _p3[2] )
        {
    /*----------------------- test node-line intersection */
            return node_line_3d (
                _pp, _p3, _p1, _qq, _bind) ;
        }
        else
        if (_p3[0] == _p1[0] &&
            _p3[1] == _p1[1] &&
            _p3[2] == _p1[2] )
        {
    /*----------------------- test node-line intersection */
            return node_line_3d (
                _pp, _p1, _p2, _qq, _bind) ;
        }
        else
        {
    /*----------------------- test node-tria intersection */
            double _PP[3] ;
            _PP[0] =  _pp[0] ;
            _PP[1] =  _pp[1] ;
            _PP[2] =  _pp[2] ;

            double _P1[3] ;
            _P1[0] =  _p1[0] ;
            _P1[1] =  _p1[1] ;
            _P1[2] =  _p1[2] ;

            double _P2[3] ;
            _P2[0] =  _p2[0] ;
            _P2[1] =  _p2[1] ;
            _P2[2] =  _p2[2] ;

            double _P3[3] ;
            _P3[0] =  _p3[0] ;
            _P3[1] =  _p3[1] ;
            _P3[2] =  _p3[2] ;

            double _ss = 
                geompred::orient3d (
                (double*)_P1 , 
                (double*)_P2 , 
                (double*)_P3 , 
                (double*)_PP )  ;

            if (_ss == +0.0)
            {
    /*----------------------- get orientation in xy-plane */
            double _TP[2] ;
            _TP[0] =  _PP[0] ;
            _TP[1] =  _PP[1] ;

            double _T1[2] ;
            _T1[0] =  _P1[0] ;
            _T1[1] =  _P1[1] ;

            double _T2[2] ;
            _T2[0] =  _P2[0] ;
            _T2[1] =  _P2[1] ;

            double _T3[2] ;
            _T3[0] =  _P3[0] ;
            _T3[1] =  _P3[1] ;
            
            double _s3[3] ;
            _s3[0] = geompred::orient2d (
                (double*)_T1 , 
                (double*)_T2 , 
                (double*)_TP )  ;

            double _s1[3] ;
            _s1[0] = geompred::orient2d (
                (double*)_T2 , 
                (double*)_T3 , 
                (double*)_TP )  ;

            double _s2[3] ;
            _s2[0] = geompred::orient2d (
                (double*)_T3 , 
                (double*)_T1 , 
                (double*)_TP )  ;

            if (_s1[0]*_s2[0] < +0.0 || 
                _s2[0]*_s3[0] < +0.0 || 
                _s3[0]*_s1[0] < +0.0 )
            {
    /*----------------------- no intersections: null hits */
            return null_hits ;
            }

    /*----------------------- get orientation in xz-plane */
            _TP[0] =  _PP[0] ;
            _TP[1] =  _PP[2] ;

            _T1[0] =  _P1[0] ;
            _T1[1] =  _P1[2] ;

            _T2[0] =  _P2[0] ;
            _T2[1] =  _P2[2] ;

            _T3[0] =  _P3[0] ;
            _T3[1] =  _P3[2] ;
            
            _s3[1] = geompred::orient2d (
                (double*)_T1 , 
                (double*)_T2 , 
                (double*)_TP )  ;

            _s1[1] = geompred::orient2d (
                (double*)_T2 , 
                (double*)_T3 , 
                (double*)_TP )  ;

            _s2[1] = geompred::orient2d (
                (double*)_T3 , 
                (double*)_T1 , 
                (double*)_TP )  ;

            if (_s1[1]*_s2[1] < +0.0 || 
                _s2[1]*_s3[1] < +0.0 || 
                _s3[1]*_s1[1] < +0.0 )
            {
    /*----------------------- no intersections: null hits */
            return null_hits ;
            }

    /*----------------------- get orientation in yz-plane */
            _TP[0] =  _PP[1] ;
            _TP[1] =  _PP[2] ;

            _T1[0] =  _P1[1] ;
            _T1[1] =  _P1[2] ;

            _T2[0] =  _P2[1] ;
            _T2[1] =  _P2[2] ;

            _T3[0] =  _P3[1] ;
            _T3[1] =  _P3[2] ;
            
            _s3[2] = geompred::orient2d (
                (double*)_T1 , 
                (double*)_T2 , 
                (double*)_TP )  ;

            _s1[2] = geompred::orient2d (
                (double*)_T2 , 
                (double*)_T3 , 
                (double*)_TP )  ;

            _s2[2] = geompred::orient2d (
                (double*)_T3 , 
                (double*)_T1 , 
                (double*)_TP )  ;

            if (_s1[2]*_s2[2] < +0.0 || 
                _s2[2]*_s3[2] < +0.0 || 
                _s3[2]*_s1[2] < +0.0 )
            {
    /*----------------------- no intersections: null hits */
            return null_hits ;
            }

    /*----------------------- test intersection hierarchy */
            bool_type _z1 = 
                _s1[0] == +0.0 &&
                _s1[1] == +0.0 &&
                _s1[2] == +0.0  ;

            bool_type _z2 = 
                _s2[0] == +0.0 &&
                _s2[1] == +0.0 &&
                _s2[2] == +0.0  ;

            bool_type _z3 = 
                _s3[0] == +0.0 &&
                _s3[1] == +0.0 &&
                _s3[2] == +0.0  ;

            if (_z1 && _z2 && _z3 )
            {
    /*----------------------- degenerate tria: check line */

        //!!std::cout << "node-tria-3d" << std::endl;

            return null_hits ;

            }
            else
            if (_z1 && _z2 )
            {
    /*----------------------- have node-node intersection */
            _qq[0] =  _pp[0] ;
            _qq[1] =  _pp[1] ;
            _qq[2] =  _pp[2] ;

            return edge_hits ;
            }
            else
            if (_z2 && _z3 )
            {
    /*----------------------- have node-node intersection */
            _qq[0] =  _pp[0] ;
            _qq[1] =  _pp[1] ;
            _qq[2] =  _pp[2] ;

            return node_hits ;
            }
            else
            if (_z3 && _z1 )
            {
    /*----------------------- have node-node intersection */
            _qq[0] =  _pp[0] ;
            _qq[1] =  _pp[1] ;
            _qq[2] =  _pp[2] ;

            return node_hits ;
            }
            else
            if (_z1 )
            {
    /*----------------------- have node-edge intersection */
            _qq[0] =  _pp[0] ;
            _qq[1] =  _pp[1] ;
            _qq[2] =  _pp[2] ;

            return edge_hits ;
            }
            else
            if (_z2 )
            {
    /*----------------------- have node-edge intersection */
            _qq[0] =  _pp[0] ;
            _qq[1] =  _pp[1] ;
            _qq[2] =  _pp[2] ;

            return edge_hits ;
            }
            else
            if (_z3 )
            {
    /*----------------------- have node-edge intersection */
            _qq[0] =  _pp[0] ;
            _qq[1] =  _pp[1] ;
            _qq[2] =  _pp[2] ;

            return edge_hits ;
            }
            else
            {
    /*----------------------- have node-tria intersection */
            _qq[0] =  _pp[0] ;
            _qq[1] =  _pp[1] ;
            _qq[2] =  _pp[2] ;

            return face_hits ;
            }

            }
            else
            {
    /*----------------------- no intersections: null hits */
            return null_hits ;
            }
        }
    }

    /*
    --------------------------------------------------------
     * LINE-TRIA-KD: line-tria intersections
    --------------------------------------------------------
     */

    template <
        typename  real_type
             >
    __normal_call hits_type line_tria_3d (
    __const_ptr  (real_type) _pa, // line
    __const_ptr  (real_type) _pb,
    __const_ptr  (real_type) _p1, // tria
    __const_ptr  (real_type) _p2,
    __const_ptr  (real_type) _p3,
    __write_ptr  (real_type) _qq, // intersection
                  bool_type _bind = true ,
                  char_type _part = +1
        )
    {

        __unreferenced (_part);

        if (_pa[0] == _pb[0] &&
            _pa[1] == _pb[1] &&
            _pa[2] == _pb[2] )
        {
    /*----------------------- test node-tria intersection */
            return node_tria_3d (
                _pa, _p1, _p2, _p3, _qq, _bind) ;
        }
        else
        if (_p1[0] == _p2[0] &&
            _p1[1] == _p2[1] &&
            _p1[2] == _p2[2] )
        {
    /*----------------------- test line-line intersection */
            return line_line_3d (
            _pa, _pb, _p2, _p3, _qq, _bind) ;
        }
        else
        if (_p2[0] == _p3[0] &&
            _p2[1] == _p3[1] &&
            _p2[2] == _p3[2] )
        {
    /*----------------------- test line-line intersection */
            return line_line_3d (
            _pa, _pb, _p3, _p1, _qq, _bind) ;
        }
        else
        if (_p3[0] == _p1[0] &&
            _p3[1] == _p1[1] &&
            _p3[2] == _p1[2] )
        {
    /*----------------------- test line-line intersection */
            return line_line_3d (
            _pa, _pb, _p1, _p2, _qq, _bind) ;
        }
        else
        {
    /*----------------------- test line-tria intersection */
            double _PA[3] ;
            _PA[0] =  _pa[0] ;
            _PA[1] =  _pa[1] ;
            _PA[2] =  _pa[2] ;

            double _PB[3] ;
            _PB[0] =  _pb[0] ;
            _PB[1] =  _pb[1] ;
            _PB[2] =  _pb[2] ;

            double _P1[3] ;
            _P1[0] =  _p1[0] ;
            _P1[1] =  _p1[1] ;
            _P1[2] =  _p1[2] ;

            double _P2[3] ;
            _P2[0] =  _p2[0] ;
            _P2[1] =  _p2[1] ;
            _P2[2] =  _p2[2] ;

            double _P3[3] ;
            _P3[0] =  _p3[0] ;
            _P3[1] =  _p3[1] ;
            _P3[2] =  _p3[2] ;

    /*----------------------- test if line straddles tria */
            double _sa = 
                geompred::orient3d (
                (double*)_P1 , 
                (double*)_P2 , 
                (double*)_P3 , 
                (double*)_PA )   ;
        
            double _sb = 
                geompred::orient3d (
                (double*)_P1 , 
                (double*)_P3 , 
                (double*)_P2 , 
                (double*)_PB )   ;

            if (_bind)
            {
    /*----------------------- no intersections: null hits */
            if (_sa*_sb < 0.0)
            return null_hits ;
            }

    /*----------------------- test if tria straddles line */
            double _s1 = 
                geompred::orient3d (
                (double*)_P1 , 
                (double*)_P2 ,
                (double*)_PA , 
                (double*)_PB )   ;

            double _s2 = 
                geompred::orient3d (
                (double*)_P2 , 
                (double*)_P3 ,
                (double*)_PA , 
                (double*)_PB )   ;
        
            double _s3 = 
                geompred::orient3d (
                (double*)_P3 , 
                (double*)_P1 ,
                (double*)_PA , 
                (double*)_PB )   ;

            if (_bind)
            {
    /*----------------------- no intersections: null hits */
            if (_s1*_s2 < 0.0)
            return null_hits ;

            if (_s2*_s3 < 0.0)
            return null_hits ;
            
            if (_s3*_s1 < 0.0)
            return null_hits ;
            }

            if (_sa == +0.0 &&
                _sb == +0.0 )
            {
        // line + tria in same plane

        //!!std::cout << "line-tria-3d" << std::endl;

            return null_hits ;

            }
            else
            if (_s1 == +0.0 &&
                _s2 == +0.0 )
            {
    /*----------------------- have line-node intersection */
            _qq[0] =  _p2[0] ;
            _qq[1] =  _p2[1] ;
            _qq[2] =  _p2[2] ;

            return node_hits ;
            }
            else
            if (_s2 == +0.0 &&
                _s3 == +0.0 )
            {
    /*----------------------- have line-node intersection */
            _qq[0] =  _p3[0] ;
            _qq[1] =  _p3[1] ;
            _qq[2] =  _p3[2] ;

            return node_hits ;
            }
            else
            if (_s3 == +0.0 &&
                _s1 == +0.0 )
            {
    /*----------------------- have line-node intersection */
            _qq[0] =  _p1[0] ;
            _qq[1] =  _p1[1] ;
            _qq[2] =  _p1[2] ;

            return node_hits ;
            }
            else
            if (_s1 == +0.0 )
            {
    /*----------------------- have line-edge intersection */            
            double _WS = _s2 + _s3 ;
            double _W1 = _s2 / _WS ;
            double _W2 = _s3 / _WS ;
            
            dd_flt _F1[3] ;
            _F1[0] =  _p1[0] ;
            _F1[1] =  _p1[1] ;
            _F1[2] =  _p1[2] ;

            dd_flt _F2[3] ;
            _F2[0] =  _p2[0] ;
            _F2[1] =  _p2[1] ;
            _F2[2] =  _p2[2] ;

            dd_flt _QQ[3] ;
            _QQ[0] =  _F1[0] * _W1 +
                      _F2[0] * _W2 ;
            _QQ[1] =  _F1[1] * _W1 +
                      _F2[1] * _W2 ;
            _QQ[2] =  _F1[2] * _W1 +
                      _F2[2] * _W2 ;

            _qq[0] =  _QQ[0] ;
            _qq[1] =  _QQ[1] ;
            _qq[2] =  _QQ[2] ;

            return edge_hits ;
            }
            else
            if (_s2 == +0.0 )
            {
    /*----------------------- have line-edge intersection */            
            double _WS = _s1 + _s3 ;
            double _W2 = _s3 / _WS ;
            double _W3 = _s1 / _WS ;
            
            dd_flt _F2[3] ;
            _F2[0] =  _p2[0] ;
            _F2[1] =  _p2[1] ;
            _F2[2] =  _p2[2] ;

            dd_flt _F3[3] ;
            _F3[0] =  _p3[0] ;
            _F3[1] =  _p3[1] ;
            _F3[2] =  _p3[2] ;

            dd_flt _QQ[3] ;
            _QQ[0] =  _F2[0] * _W2 +
                      _F3[0] * _W3 ;
            _QQ[1] =  _F2[1] * _W2 +
                      _F3[1] * _W3 ;
            _QQ[2] =  _F2[2] * _W2 +
                      _F3[2] * _W3 ;

            _qq[0] =  _QQ[0] ;
            _qq[1] =  _QQ[1] ;
            _qq[2] =  _QQ[2] ;

            return edge_hits ;
            }
            else
            if (_s3 == +0.0 )
            {
    /*----------------------- have line-edge intersection */            
            double _WS = _s1 + _s2 ;
            double _W3 = _s1 / _WS ;
            double _W1 = _s2 / _WS ;
            
            dd_flt _F3[3] ;
            _F3[0] =  _p3[0] ;
            _F3[1] =  _p3[1] ;
            _F3[2] =  _p3[2] ;

            dd_flt _F1[3] ;
            _F1[0] =  _p1[0] ;
            _F1[1] =  _p1[1] ;
            _F1[2] =  _p1[2] ;

            dd_flt _QQ[3] ;
            _QQ[0] =  _F3[0] * _W3 +
                      _F1[0] * _W1 ;
            _QQ[1] =  _F3[1] * _W3 +
                      _F1[1] * _W1 ;
            _QQ[2] =  _F3[2] * _W3 +
                      _F1[2] * _W1 ;

            _qq[0] =  _QQ[0] ;
            _qq[1] =  _QQ[1] ;
            _qq[2] =  _QQ[2] ;

            return edge_hits ;
            }
            else
            {
    /*----------------------- have line-face intersection */            
            double _WS = _s1 + _s2 + _s3 ;
            double _W1 = _s2 / _WS ;
            double _W2 = _s3 / _WS ;
            double _W3 = _s1 / _WS ;
            
            dd_flt _F1[3] ;
            _F1[0] =  _p1[0] ;
            _F1[1] =  _p1[1] ;
            _F1[2] =  _p1[2] ;

            dd_flt _F2[3] ;
            _F2[0] =  _p2[0] ;
            _F2[1] =  _p2[1] ;
            _F2[2] =  _p2[2] ;

            dd_flt _F3[3] ;
            _F3[0] =  _p3[0] ;
            _F3[1] =  _p3[1] ;
            _F3[2] =  _p3[2] ;

            dd_flt _QQ[3] ;
            _QQ[0] =  _F1[0] * _W1 +
                      _F2[0] * _W2 +
                      _F3[0] * _W3 ;
            _QQ[1] =  _F1[1] * _W1 +
                      _F2[1] * _W2 +
                      _F3[1] * _W3 ;
            _QQ[2] =  _F1[2] * _W1 +
                      _F2[2] * _W2 +
                      _F3[2] * _W3 ;

            _qq[0] =  _QQ[0] ;
            _qq[1] =  _QQ[1] ;
            _qq[2] =  _QQ[2] ;

            return face_hits ;
            }
        }
    }


    }

#   endif   //__INTERSECT_K__



