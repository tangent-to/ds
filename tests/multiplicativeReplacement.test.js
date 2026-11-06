import { describe, expect, it } from "vitest";
import { closure, multiplicativeReplacement, clr } from "../src/mva/composition.js";

const sum = (r) => r.reduce((a, b) => a + b, 0);

describe("multiplicativeReplacement", () => {
  // Seven parties, two with no vote; the 0.008 share fell below the old
  // formula's uniform subtraction (delta·2/5 ≈ 0.0082) and went negative.
  const row = [0, 0, 0.45, 0.038, 0.47, 0.008, 0.033];
  const delta = (1 / 7) ** 2;

  it("keeps every part positive and the row closed", () => {
    const r = multiplicativeReplacement(row, delta);
    expect(r).toHaveLength(7);
    expect(r.every((v) => v > 0)).toBe(true);
    expect(sum(r)).toBeCloseTo(1, 12);
  });

  it("follows the Martín-Fernández et al. (2003) formula", () => {
    const r = multiplicativeReplacement(row, delta);
    const c = closure(row);
    expect(r[0]).toBeCloseTo(delta, 12);
    expect(r[1]).toBeCloseTo(delta, 12);
    for (const j of [2, 3, 4, 5, 6]) {
      expect(r[j]).toBeCloseTo(c[j] * (1 - 2 * delta), 12);
    }
  });

  it("preserves the ratios between non-zero parts", () => {
    const r = multiplicativeReplacement(row, delta);
    expect(Math.abs(r[2] / r[4] - row[2] / row[4])).toBeLessThan(1e-12);
    expect(Math.abs(r[3] / r[5] - row[3] / row[5])).toBeLessThan(1e-12);
    expect(Math.abs(r[5] / r[6] - row[5] / row[6])).toBeLessThan(1e-12);
  });

  it("only closes a row with no zeros", () => {
    const x = [2, 3, 5];
    const r = multiplicativeReplacement(x, 0.01);
    const c = closure(x);
    r.forEach((v, j) => expect(v).toBeCloseTo(c[j], 15));
  });

  it("closes the row before replacing", () => {
    const raw = [0, 0.2, 0.48, 0.3]; // sums to 0.98
    const a = multiplicativeReplacement(raw, 0.01);
    const b = multiplicativeReplacement(closure(raw), 0.01);
    a.forEach((v, j) => expect(Math.abs(v - b[j])).toBeLessThan(1e-12));
    expect(a[0]).toBeCloseTo(0.01, 12);
    // and scaling the counts changes nothing either
    const counts = multiplicativeReplacement(raw.map((v) => v * 1000), 0.01);
    counts.forEach((v, j) => expect(Math.abs(v - a[j])).toBeLessThan(1e-12));
  });

  it("accepts one threshold per part", () => {
    const x = [0, 0.5, 0, 0.5];
    const d = [0.01, 0.9, 0.03, 0.9]; // thresholds of non-zero parts are unused
    const r = multiplicativeReplacement(x, d);
    expect(r[0]).toBeCloseTo(0.01, 12);
    expect(r[2]).toBeCloseTo(0.03, 12);
    expect(r[1]).toBeCloseTo(0.5 * (1 - 0.04), 12);
    expect(r[3]).toBeCloseTo(0.5 * (1 - 0.04), 12);
    expect(sum(r)).toBeCloseTo(1, 12);
  });

  it("rejects negative values", () => {
    expect(() => multiplicativeReplacement([0.5, -0.1, 0.6], 0.01)).toThrow(/negative/);
  });

  it("rejects a row with a zero total, naming it", () => {
    expect(() => multiplicativeReplacement([[1, 0, 1], [0, 0, 0]], 0.01)).toThrow(
      /row 1 has a zero total/,
    );
  });

  it("rejects thresholds that reach the row total, naming the row", () => {
    expect(() => multiplicativeReplacement([[1, 2, 3], [0, 0, 1]], 0.5)).toThrow(/row 1/);
    expect(() => multiplicativeReplacement([[0, 0.5, 0.5]], [1, 0.1, 0.1])).toThrow(/row 0/);
    // just below is fine
    expect(() => multiplicativeReplacement([0, 0, 1], 0.49)).not.toThrow();
  });

  it("rejects a delta array of the wrong length, or a non-positive delta", () => {
    expect(() => multiplicativeReplacement([0, 1, 1], [0.01, 0.01])).toThrow(/3 parts/);
    expect(() => multiplicativeReplacement([0, 1, 1], 0)).toThrow(/positive/);
  });

  it("keeps the shape of the input: matrix and vector", () => {
    const M = [row, [0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1], [0, 1, 1, 1, 1, 1, 1]];
    const R = multiplicativeReplacement(M, delta);
    expect(R).toHaveLength(3);
    R.forEach((r, i) => {
      expect(Array.isArray(r)).toBe(true);
      expect(r).toHaveLength(7);
      expect(sum(r)).toBeCloseTo(1, 12);
      expect(r.every((v) => v > 0)).toBe(true);
      // row by row, the same as the vector call
      const v = multiplicativeReplacement(M[i], delta);
      v.forEach((x, j) => expect(x).toBe(r[j]));
    });
    const v = multiplicativeReplacement(row, delta);
    expect(Array.isArray(v[0])).toBe(false);
  });

  it("does not modify its input", () => {
    const x = row.slice();
    multiplicativeReplacement(x, delta);
    expect(x).toEqual(row);
  });

  it("lets clr handle the zeros it used to choke on", () => {
    const out = clr([row], true, delta);
    expect(out[0].every(Number.isFinite)).toBe(true);
  });
});
