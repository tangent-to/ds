import { describe, expect, it } from 'vitest';
import {
  LabelEncoder,
  MinMaxScaler,
  OneHotEncoder,
  PolynomialFeatures,
  StandardScaler,
} from '../src/ml/preprocessing.js';

describe('Preprocessing', () => {
  describe('StandardScaler', () => {
    it('should scale to zero mean and unit variance', () => {
      const scaler = new StandardScaler();
      const X = [[1, 2], [3, 4], [5, 6]];

      scaler.fit(X);
      const Xscaled = scaler.transform(X);

      expect(Xscaled.length).toBe(3);
      expect(scaler.means.length).toBe(2);
    });

    it('should accept table descriptors and return scaled rows', () => {
      const scaler = new StandardScaler();
      const data = [
        { a: 1, b: 10, species: 'x' },
        { a: 2, b: 20, species: 'y' },
        { a: 3, b: 30, species: 'z' },
      ];

      scaler.fit({ data, columns: ['a', 'b'] });
      const result = scaler.transform({ data, columns: ['a', 'b'] });

      expect(result.columns).toEqual(['a', 'b']);
      expect(result.data).toHaveLength(3);
      expect(result.data[0].species).toBe('x');
      expect(Array.isArray(result.X)).toBe(true);
    });
  });

  describe('MinMaxScaler', () => {
    it('should scale to [0, 1]', () => {
      const scaler = new MinMaxScaler();
      const X = [[1], [2], [3]];

      scaler.fit(X);
      const Xscaled = scaler.transform(X);

      expect(Xscaled[0][0]).toBe(0);
      expect(Xscaled[2][0]).toBe(1);
    });
  });

  describe('LabelEncoder', () => {
    it('should encode labels to indices', () => {
      const encoder = new LabelEncoder();
      const y = ['cat', 'dog', 'cat', 'bird'];

      encoder.fit(y);
      const yEncoded = encoder.transform(y);

      expect(yEncoded.length).toBe(4);
      expect(yEncoded[0]).toBe(yEncoded[2]); // Both 'cat'
    });

    it('should accept table descriptors and return encoded rows', () => {
      const encoder = new LabelEncoder();
      const data = [
        { species: 'setosa', value: 1 },
        { species: 'versicolor', value: 2 },
        { species: 'setosa', value: 3 },
      ];

      encoder.fit({ data, column: 'species' });
      const encoded = encoder.transform({ data, column: 'species' });

      expect(Array.isArray(encoded.values)).toBe(true);
      expect(encoded.values[0]).toBe(encoder.classMap.get('setosa'));
      expect(encoded.data[0].species).toBe(encoded.values[0]);
    });
  });

  describe('OneHotEncoder', () => {
    it('should one-hot encode categories', () => {
      const encoder = new OneHotEncoder();
      const X = [['A'], ['B'], ['A']];

      encoder.fit(X);
      const Xencoded = encoder.transform(X);

      expect(Xencoded[0]).toEqual([1, 0]);
      expect(Xencoded[1]).toEqual([0, 1]);
    });

    it('should work with table descriptors and append encoded columns', () => {
      const encoder = new OneHotEncoder();
      const data = [
        { color: 'red', size: 'S' },
        { color: 'blue', size: 'M' },
        { color: 'red', size: 'L' },
      ];

      encoder.fit({ data, columns: ['color'] });
      const encoded = encoder.transform({ data, columns: ['color'] });

      expect(encoded.columns).toEqual(['color_blue', 'color_red']);
      expect(encoded.data[0].color_red).toBe(1);
      expect(encoded.data[1].color_blue).toBe(1);
    });
  });

  describe('PolynomialFeatures', () => {
    it('should expand table inputs and expose transformed rows', () => {
      const poly = new PolynomialFeatures({ degree: 2, includeBias: true });
      const data = [
        { x: 1, y: 2 },
        { x: 3, y: 4 },
      ];

      const transformed = poly.fitTransform({ data, columns: ['x', 'y'] });

      expect(transformed.columns).toEqual(['bias', 'x', 'y', 'x^2', 'x*y', 'y^2']);
      expect(transformed.data[0].x).toBe(1);
      expect(transformed.data[0]['x^2']).toBe(1);
      expect(transformed.data[0]['x*y']).toBe(2);
      expect(transformed.data[1]['y^2']).toBe(16);
    });
  });
});
