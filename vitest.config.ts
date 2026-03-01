
import { defineConfig } from 'vitest/config';





export default defineConfig({

  test: {

    globals: true,
    include: ['src/ts/**/*.spec.ts'],
    exclude: ['dist/**', 'node_modules/**'],

    coverage: {
      include: ['src/ts/**/*.ts'],
      exclude: ['src/ts/**/*.spec.ts'],
    }

  }

});
