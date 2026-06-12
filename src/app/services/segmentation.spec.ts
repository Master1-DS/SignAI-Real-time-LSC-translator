import { TestBed } from '@angular/core/testing';

import { Segmentation } from './segmentation';

describe('Segmentation', () => {
  let service: Segmentation;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(Segmentation);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
