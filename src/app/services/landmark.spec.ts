import { TestBed } from '@angular/core/testing';

import { Landmark } from './landmark';

describe('Landmark', () => {
  let service: Landmark;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(Landmark);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
