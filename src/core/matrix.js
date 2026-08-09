/**
 * Minimal dense matrix class over plain nested arrays.
 *
 * Replicates the subset of the ml-matrix API used across tangent/ds so
 * the rest of the library is backend-agnostic; all decompositions live
 * in core/linalg.js on top of @tangent.to/lina. Arithmetic methods
 * (add, sub, mul, div, and the element-wise math ops) mutate in place
 * and return `this`, matching ml-matrix semantics.
 */

import { matmul } from '@tangent.to/lina';

/**
 * Extract a nested number[][] from a Matrix or pass a nested array through.
 * @param {Matrix|Array<Array<number>>} value - Matrix or nested array
 * @returns {Array<Array<number>>} Nested row-major array (not copied)
 */
export function asArray(value) {
  return value instanceof Matrix ? value.data : value;
}

export class Matrix {
  /**
   * @param {number|Array<Array<number>>|Matrix} rowsOrData - Row count,
   *   nested array, or Matrix to copy
   * @param {number} [columns] - Column count when rowsOrData is a number
   */
  constructor(rowsOrData, columns) {
    if (typeof rowsOrData === 'number') {
      this.data = Array.from({ length: rowsOrData }, () => new Array(columns).fill(0));
    } else if (rowsOrData instanceof Matrix) {
      this.data = rowsOrData.data.map((row) => row.slice());
    } else if (Array.isArray(rowsOrData)) {
      this.data = rowsOrData.map((row) => row.slice());
    } else {
      throw new TypeError('Matrix: expected dimensions, nested array, or Matrix');
    }
  }

  static zeros(rows, columns) {
    return new Matrix(rows, columns);
  }

  static ones(rows, columns) {
    const m = new Matrix(rows, columns);
    for (const row of m.data) row.fill(1);
    return m;
  }

  static eye(rows, columns = rows) {
    const m = new Matrix(rows, columns);
    for (let i = 0; i < Math.min(rows, columns); i++) m.data[i][i] = 1;
    return m;
  }

  static diag(values) {
    const n = values.length;
    const m = new Matrix(n, n);
    for (let i = 0; i < n; i++) m.data[i][i] = values[i];
    return m;
  }

  static columnVector(values) {
    return new Matrix(Array.from(values, (v) => [v]));
  }

  static rowVector(values) {
    return new Matrix([Array.from(values)]);
  }

  get rows() {
    return this.data.length;
  }

  get columns() {
    return this.data.length === 0 ? 0 : this.data[0].length;
  }

  get(i, j) {
    return this.data[i][j];
  }

  set(i, j, value) {
    this.data[i][j] = value;
    return this;
  }

  getRow(i) {
    return this.data[i].slice();
  }

  getColumn(j) {
    return this.data.map((row) => row[j]);
  }

  setRow(i, values) {
    this.data[i] = Array.from(values);
    return this;
  }

  setColumn(j, values) {
    for (let i = 0; i < this.data.length; i++) this.data[i][j] = values[i];
    return this;
  }

  to2DArray() {
    return this.data.map((row) => row.slice());
  }

  to1DArray() {
    return this.data.flat();
  }

  clone() {
    return new Matrix(this);
  }

  /**
   * Matrix product; returns a new Matrix.
   * @param {Matrix|Array<Array<number>>} other - Right operand
   * @returns {Matrix} this * other
   */
  mmul(other) {
    const result = new Matrix(0, 0);
    result.data = matmul(this.data, asArray(other));
    return result;
  }

  transpose() {
    const rows = this.rows;
    const columns = this.columns;
    const out = new Matrix(columns, rows);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < columns; j++) out.data[j][i] = this.data[i][j];
    }
    return out;
  }

  subMatrix(startRow, endRow, startColumn, endColumn) {
    const out = [];
    for (let i = startRow; i <= endRow; i++) {
      out.push(this.data[i].slice(startColumn, endColumn + 1));
    }
    return new Matrix(out);
  }

  _elementWise(other, op) {
    if (other instanceof Matrix || Array.isArray(other)) {
      const b = asArray(other);
      for (let i = 0; i < this.data.length; i++) {
        const row = this.data[i];
        for (let j = 0; j < row.length; j++) row[j] = op(row[j], b[i][j]);
      }
    } else {
      for (const row of this.data) {
        for (let j = 0; j < row.length; j++) row[j] = op(row[j], other);
      }
    }
    return this;
  }

  add(other) {
    return this._elementWise(other, (a, b) => a + b);
  }

  sub(other) {
    return this._elementWise(other, (a, b) => a - b);
  }

  mul(other) {
    return this._elementWise(other, (a, b) => a * b);
  }

  div(other) {
    return this._elementWise(other, (a, b) => a / b);
  }

  /**
   * Mean of all entries, or per-row/per-column means.
   * @param {'row'|'column'} [by] - Aggregation axis
   * @returns {number|Array<number>} Grand mean, or one mean per row/column
   */
  mean(by) {
    if (by === 'column') {
      const means = new Array(this.columns).fill(0);
      for (const row of this.data) {
        for (let j = 0; j < row.length; j++) means[j] += row[j];
      }
      return means.map((s) => s / this.rows);
    }
    if (by === 'row') {
      return this.data.map((row) => row.reduce((a, b) => a + b, 0) / row.length);
    }
    let sum = 0;
    for (const row of this.data) {
      for (const v of row) sum += v;
    }
    return sum / (this.rows * this.columns);
  }

  max() {
    let m = -Infinity;
    for (const row of this.data) {
      for (const v of row) if (v > m) m = v;
    }
    return m;
  }

  min() {
    let m = Infinity;
    for (const row of this.data) {
      for (const v of row) if (v < m) m = v;
    }
    return m;
  }
}
