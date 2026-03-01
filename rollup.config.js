
import nodeResolve from '@rollup/plugin-node-resolve';
import commonjs    from '@rollup/plugin-commonjs';
// import replace     from '@rollup/plugin-replace';

import nodePolyfills from 'rollup-plugin-polyfill-node';





const index_cjs = {

  input: 'build/ts/index.js',

  output: {
    file    : 'dist/index.cjs',
    format  : 'cjs',
    exports : 'named',
    name    : 'caching_read_issue'
  },

  plugins : [

    nodePolyfills(),

    nodeResolve({
      mainFields     : ['module', 'main'],
//    browser        : true,
      extensions     : [ '.js' ],
      preferBuiltins : true
    }),

    commonjs()

    // replace({
    //   preventAssignment      : true,
    //   'process.env.NODE_ENV' : JSON.stringify( 'production' )
    // })

  ]

};





const index_iife = {

  input: 'build/ts/index.js',

  output: {
    file    : 'dist/index.iife.js',
    format  : 'iife',
    exports : 'named',
    name    : 'caching_read_issue'
  },

  plugins : [

    nodePolyfills(),

    nodeResolve({
      mainFields     : ['module', 'main'],
//    browser        : true,
      extensions     : [ '.js' ],
      preferBuiltins : true
    }),

    commonjs()

    // replace({
    //   preventAssignment      : true,
    //   'process.env.NODE_ENV' : JSON.stringify( 'production' )
    // })

  ]

};





const index_es = {

  input: 'build/ts/index.js',

  output: {
    file    : 'dist/index.mjs',
    format  : 'es',
    exports : 'named',
    name    : 'caching_read_issue'
  },

  plugins : [

    nodePolyfills(),

    nodeResolve({
      mainFields     : ['module', 'main'],
//    browser        : true,
      extensions     : [ '.js' ],
      preferBuiltins : true
    }),

    commonjs()

    // replace({
    //   preventAssignment      : true,
    //   'process.env.NODE_ENV' : JSON.stringify( 'production' )
    // })

  ]

};





export default [ index_cjs, index_iife, index_es ];
